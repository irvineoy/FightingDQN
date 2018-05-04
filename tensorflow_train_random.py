import sys
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
from KickAI import KickAI
from MacheteModified import Machete
from tensorflow_Agent import tensorflow_agent


def check_args(args):
    for i in range(argc):
        if args[i] == "-n" or args[i] == "--n" or args[i] == "--number":
            global GAME_NUM
            GAME_NUM = int(args[i + 1])


def start_game():
    manager.registerAI("tensorflow_agent", tensorflow_agent(gateway))
    manager.registerJavaAIForTrainNum(3)  # AIの数

    manager.registerJavaAIForTrain("MctsAi", 0.5)
    manager.registerJavaAIForTrain("Machete", 0.4)
    manager.registerJavaAIForTrain("GigaThunder", 0.1)
    print("Start game")

    # game = manager.createGame("ZEN", "ZEN", "tensorflow_agent", "MctsAi_ver4_nonDelay", GAME_NUM)
    game = manager.createGameForTrain("ZEN", "ZEN", "tensorflow_agent", GAME_NUM)
    manager.runGame(game)

    print("After game")
    sys.stdout.flush()


def close_gateway():
    gateway.close_callback_server()
    gateway.close()


def main_process():
    check_args(args)
    start_game()
    close_gateway()


args = sys.argv
argc = len(args)
GAME_NUM = 10000
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
                      callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point

main_process()
