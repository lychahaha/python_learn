#jh:mix-base,lazy-type
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

#
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(300, 300)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gwin = MyWidget()
    sys.exit(app.exec_())

#窗口常用
##窗口大小
w.resize(100, 100)
w.geometry() #返回QRect
w.setGeometry(x, y, width, height) #同时设置屏幕位置和宽高
##移动窗口
w.move(100, 100)
##设置标题栏相关
w.setWindowTitle('title')
w.setWindowIcon(icon)
##设置状态栏
win.statusBar().showMessage('ready')
##显示隐藏窗口
w.show()
w.hide()
##设置中心控件
win.setCentralWidget(xxx)

#窗口flag使用
w.setWindowFlags(xxx)
Qt.WindowStaysOnTopHint #总在最前

#标签
lb = QLabel()
lb.setPixmap(QPixmap('mute.png'))
lb.setText('123')
lb.adjustSize() #根据内容调整size

#按钮
pb = QPushButton('aa')
pb.clicked.connect(func)

#输入框
le = QLineEdit()
le.textChanged[str].connect(func)

#选择框
combo = QComboBox()
combo.addItem("Ubuntu")
combo.activated[str].connect(func)

#滑动条
##参数表示垂直还是水平
sld = QSlider(Qt.Horizontal)
sld.valueChanged[int].connect(func)

#单选框
cb = QCheckBox('Show title', self)
cb.toggle() #?
##参数为(state)
##state=Qt.Checked|?
cb.stateChanged.connect(func)

#进度条
pbar = QProgressBar()
pbar.setValue(5) #0~100

#日历
cal = QCalendarWidget()
cal.setGridVisible(True)
cal.selectedDate() #返回当前选上的日期
cal.clicked[QDate].connect(func)


#提示框
##reply是QMessageBox.Yes或QMessageBox.No
reply = QMessageBox.question(self, 'Message',"Are you sure to quit?", QMessageBox.Yes |QMessageBox.No, QMessageBox.No)
#对话框
text, ok = QInputDialog.getText(self, 'Input Dialog','Enter your name:')
col = QColorDialog.getColor()
font, ok = QFontDialog.getFont()
fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

#计时器
timer = QBasicTimer()
timer.start(100, self) #100ms发一次信号,事件接收者
timer.stop()
timerEvent

#图片
pixmap = QPixmap("redrock.png")

#信号
##设置连接
pb.clicked.connect(func)
sld.valueChanged[int].connect(func)#带参数
##获取信号源
pb = self.sender()
##自定义信号
class XXX(QObject):
    closeApp = pyqtSignal()
self.xxx.closeApp.connect(func)
self.xxx.closeApp.emit()


#事件
event.accept()
event.ignore()

#加菜单栏,工具栏
exitAction = QAction(icon, '&Exit', self)
exitAction.setShortcut('Ctrl+Q')
exitAction.setStatusTip('Exit application')
exitAction.triggered.connect(qApp.quit)

menubar = self.menuBar()
fileMenu = menubar.addMenu('&File')
fileMenu.addAction(exitAction)

self.toolbar = self.addToolBar('Exit')
self.toolbar.addAction(exitAction)

#布局
##水平布局
hbox = QHBoxLayout()
##垂直布局
vbox = QVBoxLayout()
##格子布局
grid = QGridLayout()
##添加组件和子布局
hbox.addWidget(pb)
vbox.addLayout(hbox)
grid.addWidget(pb, 0, 0)
grid.addWidget(pb, 0, 0, 2, 2) #后两个是行宽和列宽
##父组件设置布局
self.setLayout(vbox)

#系统托管图标
class MySystemTrayIcon(QSystemTrayIcon):
    def __init__(self, parent=None):
        super(TrayIcon, self).__init__(parent)

        self.showAction1 = QAction("显示消息1", self, triggered=self.showM)
        self.quitAction = QAction("退出", self, triggered=self.quit)

        self.menu.addAction(self.showAction1)
        self.menu.addAction(self.quitAction)
        self.menu1.setTitle("二级菜单")
        self.setContextMenu(self.menu)

        # 把鼠标点击图标的信号和槽连接
        self.activated.connect(self.shotClick)
        # 把鼠标点击弹出消息的信号和槽连接
        self.messageClicked.connect(self.shotMsgClick)

        self.setIcon(QIcon("ico.ico"))

    def shotClick(self, k):
        # 鼠标点击icon传递的信号会带有一个整形的值，1是表示单击右键，2是双击，3是单击左键，4是用鼠标中键点击
        print(k)

    def shotMsgClick(self):
        print(233)

    def showM(self):
        # 弹出消息
        self.showMessage("测试", "我是消息", self.icon)

    def quit(self):
        self.setVisible(False)
        self.parent().close()

class window(QWidget):
    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.sys_icon = TrayIcon(self)
        self.sys_icon.show()

#视频音乐播放
mp = QMediaPlayer()
content = QMediaContent(QUrl.fromLocalFile(filename))
mp.setMedia(content)
mp.play()