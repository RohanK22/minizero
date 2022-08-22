# Play stockfish on lichess

from Node import Node
import chess
from MCTS import run_mcts, select_action
from utils import show_svg
from IPython.display import SVG
from NNetwork import Network, load_network
import requests
import json
from Game import Game
import time
import chess.svg

def get_move(board: chess.Board, network: Network):
    root = Node(0, board.copy(stack=False))
    (root, max_d) = run_mcts(root, network)
    (move, action_no) = select_action(root)
    return move

def play_lichess_ai(network: Network, gameId = None, level = 1, color = 'white', token = "Bearer lip_zb85yk13lq8WGg7uwOpo"):
    if gameId == None:
        stockfish_play_url = 'https://lichess.org/api/challenge/ai'
        data = {'level': level, 'color': color}
        headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": token}
        
        x = requests.post(stockfish_play_url, json = data, headers=headers)
        x = x.json()
        print('game id', x['id'])
        gameId = x['id']
        fen = x['fen']
    
    
    game = Game()
    game.board = chess.Board(fen)
    
    
    print('https://lichess.org/api/bot/game/stream/' + gameId)
    while not game.over():
        show_svg(SVG(chess.svg.board(game.board)))
        if Game.turn(game.board) == 0:
            # sleep and get move
            time.sleep(2)
            print('ai to play')
            d = None
            state = requests.get('https://lichess.org/api/bot/game/stream/' + gameId, headers=headers, stream=True)
            iterator = state.iter_lines()
            for line in iterator:
                if line != None:    
                    d = json.loads(line)
                    if (d['type'] == 'gameFull'):
                        moves = d['state']['moves'].split()
                    elif (d['type'] == 'gameState'):
                        moves = d['moves']
                    print(moves)
                    if len(moves) > 0: # and len(moves) % 2 == 0
                        # he finished his move
                        m = chess.Move.from_uci(moves[-1])
                        if m in list(game.board.legal_moves):
                            print('move to make', m)
                            game.make_move(m)
                            print('move', m)
                            print(Game.turn(game.board))
                            iterator.close()
                            state.close()
                            break
                        else:
                            iterator.close()
                            state.close()
                            break
                    else:
                        print('shit', len(moves))
                        if (Game.turn(game.board) == 0):
                            continue          
                        else:
                            break
        else:
            print('minizero to play')
            move = get_move(game.board, network)
            print(move)
            op = requests.post('https://lichess.org/api/bot/game/' + gameId + '/move/' + move.uci(), headers=headers)
            print(op.json())
            game.make_move(move)
    #         print(game.board.peek(),' ---- v, p ', game.root_values[-1], max(game.child_visits[-1]),'max_d', max_d, sep=' ', end='\n')
    
if __name__ == "__main__":
    network = load_network()
    play_lichess_ai(network, color='white')
    