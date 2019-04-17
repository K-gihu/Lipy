from CustomButton import CustomButton
from PyQt5.QtWidgets import QWidget,QTabWidget, QMessageBox, QLineEdit,\
    QCheckBox, QComboBox, QLabel, QGridLayout, QApplication
from special_beam import special_beam
import matplotlib.pyplot as plt


class SpecialBeamWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('特殊光束计算')
        self.setGeometry(400, 200, 800, 300)

    def create_widgets(self):
        tab_widget = QTabWidget(self)
        tab_widget.setGeometry(0, 0, 800, 300)
        self.hermitteGaussianBeamPage = HermitteGaussianBeamPage()
        self.laguerreGaussianBeamPage = LaguerreGaussianBeamPage()
        self.besselBeamPage = BesselBeamPage()
        self.besselGaussianBeamPage = BesselGaussianBeamPage()
        self.airyGaussianBeamPage = AiryGaussianBeamPage()
        self.singleVortexPage = SingleVortexPage()
        self.doubleVorticesPage = DoubleVorticesPage()
        self.multiRingVorticesPage = MultiRingVorticesPage()
        self.gaussianBeamPage = GaussianBeamPage()
        self.flatToppedGaussianBeamPage = FlatToppedGaussianBeamPage()
        tab_widget.addTab(self.hermitteGaussianBeamPage, '厄米-高斯光束')
        tab_widget.addTab(self.laguerreGaussianBeamPage, '拉盖尔-高斯光束')
        tab_widget.addTab(self.besselBeamPage, '贝塞尔光束')
        tab_widget.addTab(self.besselGaussianBeamPage, '贝塞尔-高斯光束')
        tab_widget.addTab(self.airyGaussianBeamPage, '艾里-高斯光束')
        tab_widget.addTab(self.singleVortexPage, '单涡涡旋光束')
        tab_widget.addTab(self.doubleVorticesPage, '双涡涡旋光束')
        tab_widget.addTab(self.multiRingVorticesPage, '多环涡旋光束')
        tab_widget.addTab(self.gaussianBeamPage, '高斯光束')
        tab_widget.addTab(self.flatToppedGaussianBeamPage, '平顶高斯光束')


class AbstractSpecialBeamPage(QWidget, CustomButton):
    def __init__(self):
        super().__init__()
        self.beam = special_beam()
        self.ampOrInt = None
        self.ampFlag = True

    def check_and_show_intensity_image(self):
        if self.ampOrInt is None:
            msgbox = QMessageBox(text='请先计算再显示。')
            msgbox.exec()
            return
        self.beam.show_intensity_image(self.ampOrInt, self.ampFlag)
        plt.show()


class HermitteGaussianBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('厄米-高斯光束')

    def create_widgets(self):
        self.l_entry = QLineEdit('0')
        self.m_entry = QLineEdit('0')
        self.z_entry = QLineEdit('0')
        # self.n_entry = QLineEdit('1')
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        l_label = QLabel('参数(l)：')
        m_label = QLabel('参数(m)：')
        z_label = QLabel('传输距离(z)：')
        # n_label = QLabel('(n)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.l_entry, 0, 1)
        main_layout.addWidget(self.m_entry, 1, 1)
        main_layout.addWidget(self.z_entry, 2, 1)
        # main_layout.addWidget(self.n_entry, 3, 1)
        main_layout.addWidget(self.polar_checkbox, 4, 0)
        main_layout.addWidget(l_label, 0, 0)
        main_layout.addWidget(m_label, 1, 0)
        main_layout.addWidget(z_label, 2, 0)
        # main_layout.addWidget(n_label, 3, 0)
        main_layout.addWidget(cal_button, 5, 0)
        main_layout.addWidget(show_button, 5, 1)

    def calculate(self):
        l = float(self.l_entry.text())
        m = float(self.m_entry.text())
        z = float(self.z_entry.text())
        # n = float(self.n_entry.text())
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.Hermitte_GaussianBeam(l, m, z, polar=polar)


class LaguerreGaussianBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('拉盖尔-高斯光束')

    def create_widgets(self):
        self.l_entry = QLineEdit('1')
        self.p_entry = QLineEdit('0')
        self.form_combobox = QComboBox()
        self.form_combobox.insertItems(0, ('简单模式', '复杂模式'))
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        l_label = QLabel('拓扑荷数(l)：')
        p_label = QLabel('径向指数(p)：')
        form_label = QLabel('选择模式')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.l_entry, 0, 1)
        main_layout.addWidget(self.p_entry, 1, 1)
        main_layout.addWidget(self.form_combobox, 2, 1)
        main_layout.addWidget(self.polar_checkbox, 3, 0)
        main_layout.addWidget(l_label, 0, 0)
        main_layout.addWidget(p_label, 1, 0)
        main_layout.addWidget(form_label, 2, 0)
        main_layout.addWidget(cal_button, 4, 0)
        main_layout.addWidget(show_button, 4, 1)

    def calculate(self):
        l = float(self.l_entry.text())
        p = float(self.p_entry.text())
        form = self.form_combobox.currentText()
        if form == '简单模式':
            simple = True
        else:  # 复杂模式
            simple = False
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.Laguerre_GaussianBeam(l, p, simple, polar)


class BesselBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('贝塞尔光束')

    def create_widgets(self):
        self.order_entry = QLineEdit('0')
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        order_label = QLabel('拓扑荷数(order)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.order_entry, 0, 1)
        main_layout.addWidget(self.polar_checkbox, 1, 0)
        main_layout.addWidget(order_label, 0, 0)
        main_layout.addWidget(cal_button, 2, 0)
        main_layout.addWidget(show_button, 2, 1)

    def calculate(self):
        order = float(self.order_entry.text())
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.Bessel_beam(order, polar)


class BesselGaussianBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('贝塞尔-高斯光束')

    def create_widgets(self):
        self.order_entry = QLineEdit('0')
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        order_label = QLabel('拓扑荷数(order)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.order_entry, 0, 1)
        main_layout.addWidget(self.polar_checkbox, 1, 0)
        main_layout.addWidget(order_label, 0, 0)
        main_layout.addWidget(cal_button, 2, 0)
        main_layout.addWidget(show_button, 2, 1)

    def calculate(self):
        order = float(self.order_entry.text())
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.Bessel_Gaussian_beam(order, polar)


class AiryGaussianBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('艾里-高斯光束')

    def create_widgets(self):
        self.r0_entry = QLineEdit('0')
        self.a_entry = QLineEdit('0.05')
        self.b_entry = QLineEdit('0.3')
        r0_label = QLabel('艾里主环半径(r0)：')
        a_label = QLabel('决定径向传输距离的\n指数性截断因子(a)：')
        b_label = QLabel('分布因子(b)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.r0_entry, 0, 1)
        main_layout.addWidget(self.a_entry, 1, 1)
        main_layout.addWidget(self.b_entry, 2, 1)
        main_layout.addWidget(r0_label, 0, 0)
        main_layout.addWidget(a_label, 1, 0)
        main_layout.addWidget(b_label, 2, 0)
        main_layout.addWidget(cal_button, 3, 0)
        main_layout.addWidget(show_button, 3, 1)

    def calculate(self):
        r0 = float(self.r0_entry.text())
        a = float(self.a_entry.text())
        b = float(self.b_entry.text())
        self.beam = special_beam()
        self.ampOrInt = self.beam.Airy_Gaussian_beam(r0, a, b)


class SingleVortexPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('单涡涡旋光束')

    def create_widgets(self):
        self.m_entry = QLineEdit('1')
        self.xn_entry = QLineEdit('0')
        self.yn_entry = QLineEdit('0')
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        m_label = QLabel('拓扑荷数(m)：')
        xn_label = QLabel('涡旋质心坐标(xn)：')
        yn_label = QLabel('涡旋质心坐标(yn)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.m_entry, 0, 1)
        main_layout.addWidget(self.xn_entry, 1, 1)
        main_layout.addWidget(self.yn_entry, 2, 1)
        main_layout.addWidget(m_label, 0, 0)
        main_layout.addWidget(xn_label, 1, 0)
        main_layout.addWidget(yn_label, 2, 0)
        main_layout.addWidget(self.polar_checkbox, 3, 0)
        main_layout.addWidget(cal_button, 4, 0)
        main_layout.addWidget(show_button, 4, 1)

    def calculate(self):
        m = float(self.m_entry.text())
        xn = float(self.xn_entry.text())
        yn = float(self.yn_entry.text())
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.single_vorticity(m, xn, yn, polar)


class DoubleVorticesPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('双涡涡旋光束')

    def create_widgets(self):
        self.m1_entry = QLineEdit('1')
        self.m2_entry = QLineEdit('1')
        self.d1_entry = QLineEdit('1')
        self.d2_entry = QLineEdit('0')
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        m1_label = QLabel('拓扑荷数(m1)：')
        m2_label = QLabel('拓扑荷数(m2)：')
        d1_label = QLabel('涡旋质心到光轴的距离(d1)：')
        d2_label = QLabel('涡旋质心到光轴的距离(d2)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.m1_entry, 0, 1)
        main_layout.addWidget(self.m2_entry, 1, 1)
        main_layout.addWidget(self.d1_entry, 2, 1)
        main_layout.addWidget(self.d2_entry, 3, 1)
        main_layout.addWidget(m1_label, 0, 0)
        main_layout.addWidget(m2_label, 1, 0)
        main_layout.addWidget(d1_label, 2, 0)
        main_layout.addWidget(d2_label, 3, 0)
        main_layout.addWidget(self.polar_checkbox, 4, 0)
        main_layout.addWidget(cal_button, 5, 0)
        main_layout.addWidget(show_button, 5, 1)

    def calculate(self):
        m1 = float(self.m1_entry.text())
        m2 = float(self.m2_entry.text())
        d1 = float(self.d1_entry.text())
        d2 = float(self.d2_entry.text())
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.double_vortices(m1, m2, d1, d2, polar)


class MultiRingVorticesPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('多环涡旋光束')

    def create_widgets(self):
        self.m1_entry = QLineEdit('1')
        self.m2_entry = QLineEdit('10')
        m1_label = QLabel('拓扑荷数(m1)：')
        m2_label = QLabel('拓扑荷数(m2)：')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.m1_entry, 0, 1)
        main_layout.addWidget(self.m2_entry, 1, 1)
        main_layout.addWidget(m1_label, 0, 0)
        main_layout.addWidget(m2_label, 1, 0)
        main_layout.addWidget(cal_button, 2, 0)
        main_layout.addWidget(show_button, 2, 1)

    def calculate(self):
        m1 = float(self.m1_entry.text())
        m2 = float(self.m2_entry.text())
        self.beam = special_beam()
        self.ampOrInt = self.beam.multi_ring_vortices(m1, m2, False)
        self.ampFlag = False


class GaussianBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('高斯光束')

    def create_widgets(self):
        self.polar_checkbox = QCheckBox('是否使用极坐标？')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.polar_checkbox, 0, 0)
        main_layout.addWidget(cal_button, 1, 0)
        main_layout.addWidget(show_button, 1, 1)

    def calculate(self):
        polar = self.polar_checkbox.isChecked()
        self.beam = special_beam()
        self.ampOrInt = self.beam.Gaussian_beam(polar)


class FlatToppedGaussianBeamPage(AbstractSpecialBeamPage):
    def __init__(self):
        super().__init__()
        self.create_widgets()
        self.setWindowTitle('高斯光束')

    def create_widgets(self):
        self.N0_entry = QLineEdit('0')
        self.N0_label = QLabel('光束序号(N0)')
        cal_button = self.create_button('计算复振幅', self.calculate)
        show_button = self.create_button('显示强度图像', self.check_and_show_intensity_image)

        # 布局
        main_layout = QGridLayout(self)
        main_layout.addWidget(self.N0_entry, 0, 1)
        main_layout.addWidget(self.N0_label, 0, 0)
        main_layout.addWidget(cal_button, 1, 0)
        main_layout.addWidget(show_button, 1, 1)

    def calculate(self):
        N0 = int(self.N0_entry.text())
        self.beam = special_beam()
        self.ampOrInt = self.beam.Flat_topped_gaussian_beam(N0)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    test_cal = SpecialBeamWindow()
    test_cal.show()
    sys.exit(app.exec())

