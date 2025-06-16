import pygame
import chess
import chess.engine

# Setting
pygame.init()
width, height = 560, 560  # 560 *560
square_size = width // 8  # square_size 70
white = (238, 238, 210)  # white color
black = (118, 150, 86)  # black color
text_color = (200, 0, 0)

font = pygame.font.SysFont(None, 48)

screen = pygame.display.set_mode((width+280, height))
pygame.display.set_caption('Chess Game (Joon & Boaz)')
clock = pygame.time.Clock()
# Loading images: 'bp.png','bn.png',bb.'png','br.png','bq.png','bk.png','wp.png','wn.png',wb.'png','wr.png','wq.png','wk.png'
# White pieces (P,N,B.R.Q.K) black pieces (p,n,b,r,q,k)
piece_images = {}


def parser(s):  # If the first letter start with 'w', change it to capital letter else small letter
    if s[0] == 'w':
        return chr(ord(s[1])-32)
    else:
        return s[1]


for symbol in ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']:
    img = pygame.image.load(f'{symbol}.png')
    piece_images[parser(symbol)] = pygame.transform.scale(
        img, (square_size, square_size))
# lamba is the one that we can use without making any values

# chess engine initialization
engine = chess.engine.SimpleEngine.popen_uci(
    r"/opt/homebrew/Cellar/stockfish/17.1/bin/stockfish")

# python-chess Create board
board = chess.Board()

# UCI = now/to move (e2e4)
files = 'abcdefgh'


def coord_to_uci(col, row):

    # col = vertical line, row = horizontal line
    return files[col] + str(8 - row)
# (8 - row) to reverse the board


# main route
selected_sq = None  # Select nothing
running = True

while running:
    clock.tick(60)

    # Using event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Mouse click -> starting point
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            col, row = event.pos[0] // square_size, event.pos[1] // square_size
            uci_sq = coord_to_uci(col, row)
            idx = chess.square(col, 7 - row)
            piece = board.piece_at(idx)
            if piece and piece.color == board.turn:  # Can only click when it's your turn
                selected_sq = uci_sq

        # Mouse unclick -> ending point / if not possible it doesn't move
        elif event.type == pygame.MOUSEBUTTONUP and selected_sq:
            col, row = event.pos[0] // square_size, event.pos[1] // square_size
            dest_sq = coord_to_uci(col, row)
            move = None
            # 1) If staring point and ending point are same -> no movement
            if dest_sq != selected_sq:
                try:
                    move = chess.Move.from_uci(selected_sq + dest_sq)
                except ValueError:
                    # If it's a wrong point -> no movement
                    move = None

            # 2) If it's a right point -> play
            if move and move in board.legal_moves:
                board.push(move)
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)

    # Print chess board
    for r in range(8):
        for c in range(8):
            # If it's even number white else black ((r + c) % 2 == 0)
            color = white if (r + c) % 2 == 0 else black
            rect = (c * square_size, r * square_size, square_size, square_size)
            pygame.draw.rect(screen, color, rect)
    rect = (560, 0, 280, 560)
    pygame.draw.rect(screen, black, rect)

    # Pieces
    for sq_idx, piece in board.piece_map().items():
        file = chess.square_file(sq_idx)
        rank = chess.square_rank(sq_idx)
        col = file
        row = 7 - rank
        img = piece_images[piece.symbol()]
        screen.blit(img, (col * square_size, row * square_size))

    # Print "White wins!"or"Black wins!"or"Draw!"
    if board.is_game_over():
        result = board.result()  # '1-0', '0-1', '1/2-1/2'
        if result == '1-0':
            msg = "White wins!"
        elif result == '0-1':
            msg = "Black wins!"
        else:
            msg = "Draw!"

        text_surf = font.render(msg, True, text_color)
        text_rect = text_surf.get_rect(center=(width // 2, height // 2))
        # 반투명 배경 박스
        overlay = pygame.Surface(
            (text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))
        screen.blit(overlay, (text_rect.x - 10, text_rect.y - 10))
        screen.blit(text_surf, text_rect)
    pygame.display.flip()

pygame.quit()
