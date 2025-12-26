import cv2
import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

output_dir = "./images/upscaled/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# মডেল কনফিগারেশন
# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
# model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
# model_path = './weights/RealESRGAN_x4plus.pth'

# ১. মডেল আর্কিটেকচার পরিবর্তন (যদি anime_6b ব্যবহার করেন তবে num_block=6 হবে)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
model_path = './weights/RealESRGAN_x4plus_anime_6b.pth'

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0, # ০ এর বদলে ৪০০ দিলে ছবি ছোট ছোট ভাগে প্রসেস হবে, মেমোরি কম লাগবে
    tile_pad=10,
    pre_pad=0,
    half=True # যদি GPU (Nvidia) থাকে তবে True দিন
)

img = cv2.imread('./images/original/image1.jpg', cv2.IMREAD_UNCHANGED)
# output, _ = upsampler.enhance(img, outscale=4)

# আউটস্কেল ১ করে দিলে ছবির সাইজ সেম থাকবে কিন্তু AI ডিটেইলস ইমপ্রুভ করবে
output, _ = upsampler.enhance(img, outscale=1)

# নির্দিষ্ট ডিরেক্টরিতে সেভ করা
cv2.imwrite(os.path.join(output_dir, "upscaled_realesrgan.jpg"), output)
print(f"Saved to: {output_dir}upscaled_realesrgan.jpg")