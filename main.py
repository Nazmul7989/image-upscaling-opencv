import cv2
import os
import numpy as np

# ১. মডেল লোড করা (EDSR মডেল ব্যবহার করা হয়েছে যা খুব ভালো রেজাল্ট দেয়)
# মডেলটি এখান থেকে ডাউনলোড করে নিন: https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# ডিরেক্টরি চেক করা এবং তৈরি করা
output_dir = "./images/upscaled/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

path = "./weights/EDSR_x4.pb"
sr.readModel(path)
sr.setModel("edsr", 4) # এখানে ৪ গুণ বড় হবে

# ২. ইমেজ লোড করা
image = cv2.imread("./images/original/image1.jpg")

# ৩. ইমেজ আপস্কেল করা
result = sr.upsample(image)

# একটি শার্পেনিং কার্নেল তৈরি করা
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

# ছবিটিতে ফিল্টার অ্যাপ্লাই করা
sharpened_result = cv2.filter2D(result, -1, kernel)


cv2.imwrite(os.path.join(output_dir, "upscaled_sharpened.jpg"), sharpened_result)

# ৪. নির্দিষ্ট ডিরেক্টরিতে সেভ করা
# cv2.imwrite(os.path.join(output_dir, "upscaled_opencv.jpg"), result)
print(f"Saved to: {output_dir}upscaled_opencv.jpg")