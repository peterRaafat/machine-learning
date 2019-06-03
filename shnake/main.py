from env import Env
from player import Player
from scheduler import Scheduler

def main():
    shnake = Env()
    player = Player()
    player.start()
    scheduler = Scheduler([shnake.step], [shnake.get_refresh_rate()/1000])
    scheduler.run()
    while True:
        action = player.get_action()
        shnake.act(action)
main()