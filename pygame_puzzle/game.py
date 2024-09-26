import pygame
import random
import time
import cv2
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Constants
GAME_WIDTH, GAME_HEIGHT = 600, 500
TOTAL_WIDTH = GAME_WIDTH + 400  # Extra 200 pixels on each side for hand images
ROWS, COLS = 5, 6
SQUARE_SIZE = GAME_WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Create the screen
screen = pygame.display.set_mode((TOTAL_WIDTH, GAME_HEIGHT * 1.25))
pygame.display.set_caption("Number Sequence Sliding Puzzle Game")

# Generate the matrix
numbers = list(range(1, ROWS * COLS))  # Numbers from 1 to 29
random.shuffle(numbers)
numbers.append(None)  # Add the empty spot
matrix = [numbers[i * COLS:(i + 1) * COLS] for i in range(ROWS)]

selected_tile = None
move_count = 0
illegal_move_count = 0
start_time = time.time()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load and scale hand images
left_hand_img = pygame.image.load('pygame_puzzle/assets/left_hand.png')  # Make sure you have these images
right_hand_img = pygame.image.load('pygame_puzzle/assets/right_hand.png')
hand_img_height = GAME_HEIGHT // 2
left_hand_img = pygame.transform.scale(left_hand_img, (200, hand_img_height))
right_hand_img = pygame.transform.scale(right_hand_img, (200, hand_img_height))


def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(200 + col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            if matrix[row][col] is None:
                pygame.draw.rect(screen, WHITE, rect)
            else:
                # Highlight the selected tile in red
                color = RED if selected_tile == (row, col) else GRAY
                pygame.draw.rect(screen, color, rect)
                font = pygame.font.Font(None, 74)
                text = font.render(str(matrix[row][col]), True, BLACK)
                screen.blit(text, (200 + col * SQUARE_SIZE + 20, row * SQUARE_SIZE + 10))
            pygame.draw.rect(screen, BLACK, rect, 1)  # Draw the gridline


def draw_timer_and_counter():
    elapsed_time = int(time.time() - start_time)
    font = pygame.font.Font(None, 36)
    timer_text = font.render(f"Time: {elapsed_time}s", True, BLACK)
    counter_text = font.render(f"Moves: {move_count}", True, BLACK)
    illegal_moves_text = font.render(f"Illegal: {illegal_move_count}", True, BLACK)
    screen.blit(timer_text, (210, GAME_HEIGHT + 5))
    screen.blit(counter_text, (TOTAL_WIDTH - 150, GAME_HEIGHT + 5))
    screen.blit(illegal_moves_text, (TOTAL_WIDTH // 2 - 75, GAME_HEIGHT + 5))


def draw_hands(active_hand):
    screen.blit(left_hand_img, (0, (GAME_HEIGHT - hand_img_height) // 2))
    screen.blit(right_hand_img, (TOTAL_WIDTH - 200, (GAME_HEIGHT - hand_img_height) // 2))

    if active_hand == 'left':
        pygame.draw.rect(screen, GREEN, (0, (GAME_HEIGHT - hand_img_height) // 2, 200, hand_img_height), 5)
    elif active_hand == 'right':
        pygame.draw.rect(screen, GREEN, (TOTAL_WIDTH - 200, (GAME_HEIGHT - hand_img_height) // 2, 200, hand_img_height),
                         5)


def find_empty_spot():
    for row in range(ROWS):
        for col in range(COLS):
            if matrix[row][col] is None:
                return row, col
    return None


def is_valid_move(row, col):
    empty_row, empty_col = find_empty_spot()
    return True
    # return (abs(row - empty_row) == 1 and col == empty_col) or \
    #        (abs(col - empty_col) == 1 and row == empty_row)


def handle_click(row, col):
    global selected_tile, move_count, illegal_move_count

    # Check if we are selecting a tile
    if selected_tile is None:
        if matrix[row][col] is not None:  # Only select if it's not the empty spot
            selected_tile = (row, col)  # Select the tile
    else:
        # If a tile is already selected, check if the clicked spot is empty
        if matrix[row][col] is None:
            # Check if it's a valid move (must be adjacent to the empty spot)
            empty_row, empty_col = row, col
            selected_row, selected_col = selected_tile
            if is_valid_move(selected_row, selected_col):
                # Perform the move
                matrix[empty_row][empty_col], matrix[selected_row][selected_col] = matrix[selected_row][selected_col], \
                matrix[empty_row][empty_col]
                move_count += 1
                selected_tile = None  # Deselect the tile after moving
            else:
                illegal_move_count += 1  # If it's not a valid move, increment illegal moves
        else:
            # If a non-empty cell is clicked while a tile is selected, just deselect the current tile
            selected_tile = None


def check_win():
    flattened = [num for row in matrix for num in row if num is not None]
    return flattened == list(range(1, ROWS * COLS))


def get_active_hand():
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            if middle_finger_tip.y < wrist.y:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                if thumb_tip.x < wrist.x:
                    return 'right'  # Remember, the image is flipped
                else:
                    return 'left'

    return None


# Main loop
running = True
active_hand = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if 200 <= x < 200 + GAME_WIDTH and y < GAME_HEIGHT:  # Only consider clicks inside the grid
                col = (x - 200) // SQUARE_SIZE
                row = y // SQUARE_SIZE
                if 0 <= row < ROWS and 0 <= col < COLS:
                    handle_click(row, col)

    active_hand = get_active_hand()

    screen.fill(WHITE)
    draw_grid()
    draw_timer_and_counter()
    draw_hands(active_hand)
    pygame.display.flip()

    if check_win():
        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        print(f"You win!")
        print(f"Total moves: {move_count}")
        print(f"Time taken: {elapsed_time} seconds")
        print(f"Illegal moves: {illegal_move_count}")
        running = False

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()
