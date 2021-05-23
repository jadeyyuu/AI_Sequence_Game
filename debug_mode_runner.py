"""
将文件放置在sequence_runner.py所在的目录，在pycharm中打开该文件，右键 -> debug
可以开启debug模式(需要在"想要停止的位置"添加"断点", 添加方法为，在相应代码前方[竖线更左侧]点击
左键，点击后后出现一个红色点。代码在相应位置停止后，可以查看所有的变量值，也可以在所在位置进入
更底层函数，进行单步调试[每次运行一行，精确查看变量值，确保符合预期]。更多教程可在群里交流或
直接google)

Note: 要开启debug模式，先要将已有代码稍微修改一下
将game.py中Game类Run函数中的try...语句块改为:
try:
    # selected = func_timeout(self.time_limit,agent.SelectAction,args=(actions_copy, gs_copy))
    selected = agent.SelectAction(actions_copy, gs_copy)
否则当我们debug程序暂停时，会报超时错误(func_timeout会自动计时)

github提交代码时记得将以上语句再改回来(或者不提交修改后的game.py，只提交我们写的agent文件)
"""

from optparse import OptionParser

from sequence_runner import run

if __name__ == '__main__':
    msg = ""
    parser = OptionParser("")
    options, _ = parser.parse_args(['a', 'b'])
    options.red = "agents.samples.random"
    options.blue = "agents.my_agents.bfs"  # 将my_agents.mcts替换成自己的[文件夹.文件名(没有.py)]
    options.redName = "red"
    options.blueName = "blue"
    options.textgraphics = False
    options.quiet = False
    options.superQuiet = False
    options.warningTimeLimit = 1.0
    options.startRoundWarningTimeLimit = 5.0
    options.numOfWarnings = 3
    options.multipleGames = 1
    options.setRandomSeed = 90054
    options.saveGameRecord = False
    options.output = "output"
    options.saveLog = False
    options.replay = None
    options.delay = 0.1
    options.print = True  # 打印自定义的消息
    options.num_of_agent = 4
    run(options, True, msg)