import cv2
import numpy as np
import os
class Zwtqq_model:
    def __init__(self,real_width,real_height,roi_array):
        self.real_width=real_width
        self.real_height=real_height
        self.pts1 = roi_array
        self.pts2 = np.float32([[0, 0], [0, self.real_height], [self.real_width, self.real_height], [self.real_width, 0]])
        self.M = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        self.Minv = cv2.getPerspectiveTransform(self.pts2, self.pts1)
    def add_orignimg(self, img, result_img):
        result_img=cv2.warpPerspective(result_img, self.Minv, (2160, 3840))
        ret, mask = cv2.threshold(cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(result_img, result_img, mask=mask)
        img2_fg = cv2.bitwise_and(img, img, mask=mask_inv)
        orignimg = cv2.add(img1_bg, img2_fg)
        # orignimg = cv2.addWeighted(src1=img, alpha=1, src2=result_img, beta=0.9, gamma=0)

        # 修补透视变换拼接引起的黑色像素点
        # ret, inpaint_mask = cv2.threshold(cv2.cvtColor(orignimg, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV)
        # kernel = np.ones((40, 40), np.uint8)
        # ## mask的膨胀
        # inpaint_mask = cv2.dilate(inpaint_mask, kernel)
        #
        # orignimg = cv2.inpaint(orignimg, inpaint_mask, 40, cv2.INPAINT_TELEA)
        return orignimg
    def prepocess_img(self, img):
        img=cv2.warpPerspective(img, self.M, (self.real_width, self.real_height))
        return img
    def thres_img(self,roi_img):
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi_img, (5, 5), 0)
        ret, bina_img = cv2.threshold(blur, 127, 255, cv2.THRESH_TOZERO)
        ret, bina_img = cv2.threshold(bina_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bina_img
    def remove_small_hole(self,bina_img,hole_size):
        #去除小的连通区域
        contours, hierarch = cv2.findContours(bina_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < hole_size:
                cv2.drawContours(bina_img, [contours[i]], 0, 0, -1)

        #去除大联通区域中的小孔洞
        bina_img=cv2.bitwise_not(bina_img)
        contours, hierarch = cv2.findContours(bina_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < hole_size*20:
                cv2.drawContours(bina_img, [contours[i]], 0, 0, -1)
        bina_img = cv2.bitwise_not(bina_img)
        return bina_img
    def get_result(self,no_hole_img):
        if no_hole_img.max()>0:
            index=np.where(no_hole_img==255)
            min_index = index[0].min()
            return min_index
        return None
    def draw_line(self,result,mask_img):
        if result!=None:
            color=(255, 191, 0)
            label=str((self.real_height-result)/100)

            tl = round(0.005 * (mask_img.shape[0] + mask_img.shape[1]) / 2) + 1
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c1=(0,result)
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            mask_img = cv2.line(mask_img, (0, result), (mask_img.shape[1], result), color, tl)
            cv2.rectangle(mask_img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(mask_img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tl, lineType=cv2.LINE_AA)
        return mask_img
    def addmask_img(self,no_hole_img, img):
        if no_hole_img.max()>0:
            mask = cv2.cvtColor(no_hole_img, cv2.COLOR_GRAY2BGR)
            mask_index = (no_hole_img[:,:] == 255)
            mask[mask_index]=[0,0,255]
            mask_img = cv2.addWeighted(src1=img, alpha=1, src2=mask, beta=0.5, gamma=0)
            return mask_img
        return img
    def run(self,img):
        roi_img=self.prepocess_img(img)
        bina_img=self.thres_img(roi_img)
        no_hole_img=self.remove_small_hole(bina_img,hole_size=2000)
        result = self.get_result(no_hole_img)
        mask_img=self.addmask_img(no_hole_img, roi_img)
        result_img=self.draw_line(result,mask_img)
        orignimg=self.add_orignimg(img, result_img)
        return result_img,orignimg, result

if __name__=="__main__":
    roi_array = np.float32([[259, 713], [296, 3471], [1879, 3496], [1949, 703]])
    real_width, real_height=2010, 3460
    zwtqq_model=Zwtqq_model(real_width, real_height,roi_array)
    mode="video"
    if mode=="img":
        img = cv2.imread("./imgs/1.jpg", 0)
        img=zwtqq_model.run(img)
        resize_img=cv2.resize(img,(0,0),fx=0.15,fy=0.15)
        cv2.imshow("resize_img",resize_img)
        cv2.waitKey()
    elif mode=="video":
        if not os.path.exists("./output"):
            os.mkdir("./output")
        dir_num=0
        while True:
            if not os.path.exists("./output/res"+str(dir_num)):
                os.mkdir("./output/res"+str(dir_num))
                break
            else:
                dir_num+=1

        save_path1="./output/res"+str(dir_num)+"/result.mp4"
        save_path2="./output/res"+str(dir_num)+"/orign.mp4"
        cap=cv2.VideoCapture("video/zuoweitiqianqu.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # output video codec




        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer1 = cv2.VideoWriter(save_path1,fourcc , fps, (real_width, real_height))
        vid_writer2 = cv2.VideoWriter(save_path2, fourcc, fps, (w, h))
        while True:
            ret, frame=cap.read()
            if not ret:
                break
            result_img, orignimg, result = zwtqq_model.run(frame)


            vid_writer1.write(result_img)
            vid_writer2.write(orignimg)
            orignimg = cv2.resize(orignimg, (0, 0), fx=0.2, fy=0.2)
            result_img = cv2.resize(result_img, (0, 0), fx=0.2, fy=0.2)
            cv2.imshow("orign_img",orignimg)
            cv2.imshow("resize_img", result_img)
            if cv2.waitKey(1)==27:
                break
        vid_writer1.release()
        vid_writer2.release()
        cap.release()
        cv2.destroyAllWindows()