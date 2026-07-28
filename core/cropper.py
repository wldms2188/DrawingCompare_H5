import os
import cv2
 
 
class Cropper:
 
    def crop_changes(self, before_img, after_img, changes, save_dir="output/crops"):
 
        os.makedirs(save_dir, exist_ok=True)
 
        for change in changes:
 
            margin = 20
 
            x1 = max(0, change.x - margin)
            y1 = max(0, change.y - margin)
 
            x2 = min(before_img.shape[1], change.x + change.w + margin)
            y2 = min(before_img.shape[0], change.y + change.h + margin)
 
            before_crop = before_img[y1:y2, x1:x2]
            after_crop = after_img[y1:y2, x1:x2]
 
            before_path = os.path.join(
                save_dir,
                f"{change.id}_before.png"
            )
 
            after_path = os.path.join(
                save_dir,
                f"{change.id}_after.png"
            )
 
            cv2.imwrite(before_path, before_crop)
            cv2.imwrite(after_path, after_crop)
 
            change.before_image = before_crop
            change.after_image = after_crop
 
            change.before_image_path = before_path
            change.after_image_path = after_path
 
        return changes