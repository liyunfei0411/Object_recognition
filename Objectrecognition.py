from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import requests
import logging
from logging.handlers import RotatingFileHandler


# 配置日志信息
# 设置日志的记录等级
logging.basicConfig(level=logging.INFO)
# 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024*1024*100, backupCount=10)
# 创建日志记录的格式                 日志等级    输入日志信息的文件名 行数    日志信息
formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)d %(message)s')
# 为刚创建的日志记录器设置日志记录格式
file_log_handler.setFormatter(formatter)
# 为全局的日志工具对象（flask app使用的）添加日记录器
logging.getLogger().addHandler(file_log_handler)


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/bg')
def show_bg():
    return render_template("bg.html")


@app.route('/target')
def show_target():
    return render_template("target.html")


class ObjectRecognition:

    def __init__(self, bg, target, area=None):
        '''

        :param bg: 背景图
        :param target: 目标图
        :param top: 指定识别区域顶部位置
        :param bottom: 指定识别区域底部位置
        :param left: 指定识别区域左边位置
        :param right: 指定识别区域右边位置
        '''
        self.bg = bg
        self.target = target
        self.origin_target = target
        self.width = self.origin_target.shape[0]
        self.height = self.origin_target.shape[1]
        self.area = area
        if self.area is not None:
            self.left_x = area[0]
            self.left_y = area[1]
            self.right_x = area[2]
            self.right_y = area[3]

    def clip_area(self):
        if self.area is not None:
            self.bg = self.bg[self.left_y:self.right_y, self.left_x:self.right_x]
            self.target = self.target[self.left_y:self.right_y, self.left_x:self.right_x]
            cv2.imwrite("bg.jpg", self.bg)
            cv2.imwrite("target.jpg", self.target)


    def get_contours(self):
        '''

        :param bg: 背景图
        :param target: 目标图
        :return: 多余物体的位置
        '''
        grayA = cv2.cvtColor(self.bg, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype('uint8')

        thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # ret, thresh = ...

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        return cnts


    def different(self):
        '''

        :param bg: 背景图
        :param target: 目标图
        :return: 相似度
        '''
        # 获取背景图的轮廓
        thresh_bg = cv2.Canny(self.bg, 0, 256)
        thresh_bg, contoursA, hierarchy = cv2.findContours(thresh_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 获取目标图的轮廓
        thresh_target = cv2.Canny(self.target, 0, 256)
        thresh_target, contoursB, hierarchy = cv2.findContours(thresh_target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算相似度
        sumbg = np.sum(thresh_bg)
        sumtest = np.sum(thresh_target)
        diff_score = round(float(sumbg/sumtest), 3)
        print(diff_score)
        return diff_score


    def draw_min_rect_circle(self, img, cnts):  # conts = contours

        '''
        :param img: 目标图
        :param cnts: 多余物体的位置坐标
        :return: 画出多余的物体的位置矩形框和多余物体的位置
        '''

        img = np.copy(img)
        cnt_list = list()
        if self.area is not None:
            cv2.rectangle(img, (self.left_x, self.left_y), (self.right_x, self.right_y), (0, 255, 0), 2)
            cnt_list.append([self.left_x, self.left_y, self.right_x, self.right_y])
        else:
            cnt_list.append([0, 0, self.width, self.height])
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.area is not None:
                if self.left_x > 0:
                    x = self.left_x + x
                if self.left_y > 0:
                    y = self.left_y + y
            if w > 10 and h > 10:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cnt_list.append([x, y, x+w, y+h])
        return img, cnt_list


    def main(self):
        self.clip_area()
        # 判断两个图片的相似度
        diff_score = self.different()
        # 找到多余物体的位置并画出来
        if diff_score < 0.92:
            cnts = self.get_contours()
            draw_img, cnt_list = self.draw_min_rect_circle(self.origin_target, cnts)
            cv2.imshow("result", draw_img)
            cv2.imwrite("result.jpg", draw_img)
            cv2.waitKey(3000)
            return diff_score, cnt_list
        else:
            # 两个图片相同返回零
            return 0.0, None


@app.route("/check", methods=["POST"])
def check_image():
    try:
        if not request.form.get("bg"):
            return jsonify({"error": "No bg image parameter"})
        bg_url = request.form.get("bg")
        print(bg_url)
        if not request.form.get("target"):
            return jsonify({"error": "No target image parameter"})
        target_url = request.form.get("target")
        print(target_url)
    except Exception as e:
        print(e)
        app.logger.error("e")
        return jsonify({"error": "bg or target Missing parameter"})
    try:
        area = request.form.get("area")
        area = eval(area)
    except Exception as e:
        print(e)
    if not area:
        area = None
    try:
        res_bg = requests.get(bg_url)
        image_bg = res_bg.content
        buf_bg = np.asarray(bytearray(image_bg), dtype=np.uint8)
        bg = cv2.imdecode(buf_bg, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        app.logger.error("no bg file")
        return jsonify({"error": "no bg file"})
    try:
        res_target = requests.get(target_url)
        image_bg = res_target.content
        buf = np.asarray(bytearray(image_bg), dtype=np.uint8)
        target = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        app.logger.error("no target file")
        return jsonify({"error": "no target file"})
    try:
        obj_reco = ObjectRecognition(bg, target, area)
        diff_score, coordinate_list = obj_reco.main()
        diff_dict = {"diff_score": diff_score, "coordinates": coordinate_list}
        return jsonify(diff_dict)
    except Exception as e:
        print(e)
        return jsonify({"error": "not image"})


if __name__ == '__main__':

    app.run(debug=False, host="127.0.0.1", port=5001)

