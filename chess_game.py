import pygame
import chess
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 560, 560           # Board pixel size (square = 70 × 70)
SQUARE = WIDTH // 8
LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
TEXT = (200, 0, 255)
PROMOTION_PIECE = 'q'              # auto-promote pawns to queen
FILES = 'abcdefgh'

# ---------------------------------------------------------------------------
# Neural-network helpers
# ---------------------------------------------------------------------------


def board_to_vector(board: chess.Board) -> np.ndarray:
    """One-hot encode the 8×8×12 board for the network."""
    board_vector = np.zeros(64 * 12, dtype=np.float32)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            idx = piece_map[piece.symbol()]
            board_vector[sq * 12 + idx] = 1
    return board_vector


def predict_best_move(board: chess.Board, model) -> chess.Move | None:
    """Return the legal move with the highest probability."""
    if model is None:
        print("AI model not loaded, cannot predict move.")
        return None

    board_vec = board_to_vector(board).reshape((8, 8, 12))
    preds = model.predict(np.expand_dims(board_vec, 0), verbose=0)[0]

    best_move, best_prob = None, -1.0
    for mv in board.legal_moves:
        move_idx = mv.from_square * 64 + mv.to_square  # promotion ignored in index
        if move_idx < len(preds) and preds[move_idx] > best_prob:
            best_prob, best_move = preds[move_idx], mv

    if best_move is None:
        return next(iter(board.legal_moves), None)
    return best_move


# ---------------------------------------------------------------------------
# Pygame setup
# ---------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH + 280, HEIGHT))
pygame.display.set_caption("Chess Game (Joon & Boaz)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)
small_font = pygame.font.SysFont(None, 36)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def make_button_surface(ai_white: bool) -> pygame.Surface:
    label = "Play as Black" if ai_white else "Play as White"
    return font.render(label, True, (0, 0, 0), TEXT)

# Convert 'wn' → 'N', 'bp' → 'p'...


def parser(sym: str) -> str:
    return chr(ord(sym[1]) - 32) if sym[0] == 'w' else sym[1]


# ---------------------------------------------------------------------------
# Load images once
# ---------------------------------------------------------------------------
piece_images, piece_images_outline = {}, {}
for sym in ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']:
    img = pygame.image.load(f'images/{sym}.png')
    piece_images[parser(sym)] = pygame.transform.scale(img, (SQUARE, SQUARE))
    img = pygame.image.load(f'images/{sym}_outlined.png')
    piece_images_outline[parser(sym)] = pygame.transform.scale(
        img, (SQUARE, SQUARE))

# ---------------------------------------------------------------------------
# Chess back-end
# ---------------------------------------------------------------------------
board = chess.Board()


def coord_to_uci(col: int, row: int) -> str:
    """Column/row (0-7, 0-7) → algebraic coordinate like "a2"."""
    return FILES[col] + str(8 - row)


# ---------------------------------------------------------------------------
# Load model if present
# ---------------------------------------------------------------------------
model = None
try:
    model = tf.keras.models.load_model('chess_ai_model.keras')
    print("Chess AI model loaded.")
except Exception as e:
    print(f"[WARN] Couldn’t load model: {e}\nRun model_trainer.py first.")

# ---------------------------------------------------------------------------
# Game loop state
# ---------------------------------------------------------------------------
ai_plays_white = False  # AI plays Black by default
selected_sq: str | None = None  # algebraic coordinate of selected square
button_text = make_button_surface(False)
ai_text = font.render("King Molstad", True, (0, 0, 0))
me_text = font.render("player", True, (0, 0, 0))
text_rect = button_text.get_rect()
ai_text_rect = ai_text.get_rect()
me_text_rect = me_text.get_rect()
padding = 10
center_x, center_y = 700, 280
button_width = text_rect.width + padding * 2
button_height = text_rect.height + padding * 2
ai_name_width = ai_text_rect.width + padding * 2
ai_name_height = ai_text_rect.height + padding * 2
me_name_width = me_text_rect.width + padding * 2
me_name_height = me_text_rect.height + padding * 2
button_x = center_x - button_width // 2
button_y = center_y - button_height // 2
ai_x = center_x - ai_name_width // 2
me_x = center_x - me_name_width // 2

button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

ai_rect = pygame.Rect(ai_x, button_y-200, ai_name_width, ai_name_height)
me_rect = pygame.Rect(me_x, button_y+200, me_name_width, me_name_height)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
running = True
joon_mode = False
while running:
    clock.tick(60)

    # ------------------ AI move ------------------
    if not board.is_game_over():
        if (ai_plays_white and board.turn == chess.WHITE) or \
           (not ai_plays_white and board.turn == chess.BLACK):
            ai_move = predict_best_move(board, model)
            if ai_move:
                board.push(ai_move)

    # ------------------ Event handling ------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # ---- Toggle side button ----
            if button_rect.collidepoint(event.pos):
                ai_plays_white = not ai_plays_white
                selected_sq = None
                button_surf = make_button_surface(ai_plays_white)
                continue

            if ai_rect.collidepoint(event.pos):
                joon_mode = True
                img = pygame.image.load(f'images/black_sangjoonphoto.png')
                piece_images[parser('bk')] = pygame.transform.scale(
                    img, (SQUARE, SQUARE))
                img2 = pygame.image.load(f'images/white_sangjoonphoto.png')
                piece_images[parser('wk')] = pygame.transform.scale(
                    img2, (SQUARE, SQUARE))

            # ---- Board-click (select) ----
            if event.pos[0] >= WIDTH or event.pos[1] >= HEIGHT:
                continue
            col, row = event.pos[0] // SQUARE, event.pos[1] // SQUARE
            idx = chess.square(col, 7 - row)
            piece = board.piece_at(idx)
            if piece and piece.color == board.turn:
                selected_sq = coord_to_uci(col, row)

        elif event.type == pygame.MOUSEBUTTONUP and selected_sq:
            # Ignore releases off board
            if event.pos[0] >= WIDTH or event.pos[1] >= HEIGHT:
                selected_sq = None
                continue

            col, row = event.pos[0] // SQUARE, event.pos[1] // SQUARE
            dest = coord_to_uci(col, row)
            if dest != selected_sq:
                try:
                    # ---------- Pawn-promotion logic ----------
                    orig_idx = chess.parse_square(selected_sq)
                    orig_piece = board.piece_at(orig_idx)
                    promo = ''
                    if orig_piece and orig_piece.piece_type == chess.PAWN:
                        dest_idx = chess.parse_square(dest)
                        dest_rank = chess.square_rank(dest_idx)
                        if (orig_piece.color == chess.WHITE and dest_rank == 7) or \
                           (orig_piece.color == chess.BLACK and dest_rank == 0):
                            promo = PROMOTION_PIECE  # always queen

                    move_uci = selected_sq + dest + promo
                    mv = chess.Move.from_uci(move_uci)
                    if mv in board.legal_moves:
                        board.push(mv)
                except ValueError:
                    pass
            selected_sq = None

    # ------------------ Drawing ------------------
    # Draw squares
    for r in range(8):
        for c in range(8):
            pygame.draw.rect(
                screen,
                LIGHT if (r + c) % 2 == 0 else DARK,
                (c * SQUARE, r * SQUARE, SQUARE, SQUARE)
            )

    # RHS panel background
    pygame.draw.rect(screen, DARK, (WIDTH, 0, 280, HEIGHT))

    # Changing color Button
    pygame.draw.rect(screen, TEXT, button_rect, border_radius=8)
    screen.blit(button_text, (button_x + padding, button_y + padding))

    pygame.draw.rect(screen, (150, 150, 150), me_rect)
    screen.blit(me_text, (me_x + padding, button_y+200 + padding))

    pygame.draw.rect(screen, (150, 150, 150), ai_rect)
    screen.blit(ai_text, (ai_x + padding, button_y-200 + padding))

    # Pieces
    for sq, piece in board.piece_map().items():
        col, row = chess.square_file(sq), 7 - chess.square_rank(sq)
        img = piece_images_outline[piece.symbol()] if \
            selected_sq == coord_to_uci(col, row) and not joon_mode else \
            piece_images[piece.symbol()]
        screen.blit(img, (col * SQUARE, row * SQUARE))

    # Game-over banner
    if board.is_game_over():
        result = board.result()
        msg = "White wins!" if result == '1-0' else \
              "Black wins!" if result == '0-1' else "Draw!"
        txt_surf = font.render(msg, True, TEXT)
        txt_rect = txt_surf.get_rect(center=(WIDTH + 140, HEIGHT // 2))
        overlay = pygame.Surface(
            (txt_rect.w + 20, txt_rect.h + 20), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))
        screen.blit(overlay, (txt_rect.x - 10, txt_rect.y - 10))
        screen.blit(txt_surf, txt_rect)

    pygame.display.flip()

pygame.quit()
