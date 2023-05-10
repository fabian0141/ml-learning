from Test.cave import testcave
from Graphics import graphics

runProgram = 1

match runProgram:
    case 0:
        testcave.run()
    case 1:
        graphics.run()

