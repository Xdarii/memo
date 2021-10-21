from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
count = 0
done = 0
stride=250
interval=100
def splitting_image(image_,mask_,output_path_,interval=100,stride=250):
  output_image=os.path.join(output_path_,'train/image')
  output_mask=os.path.join(output_path_,'train/label')
  if not os.path.isdir(output_image):
    os.makedirs(output_image)
  if not os.path.isdir(output_mask):
    os.makedirs(output_mask)
  for i in range(0, image_.shape[0], interval):
    for j in range(0, image_.shape[1], interval):
      cropped_image = image_[j:j + stride, i:i + stride,:]  #--- Notice this part where you have to add the stride as well ---
      cropped_mask = mask_[j:j + stride, i:i + stride]  #--- Notice this part where you have to add the stride as well ---

      if cropped_mask.shape[0]>200 and cropped_mask.shape[1]>200 and np.sum(cropped_mask) >0:
        count += 1
        cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) + image_name), cropped_image)   #--- Also take note of how you would save all the cropped images by incrementing the count variable ---
        cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) + image_name), cropped_mask)   #--- Also take note of how you would save all the cropped images by incrementing the count variable ---
def rotate_image(image, angle,scale =1.0):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
  print(image.shape[1::-1])
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def mscale(image, scale=8):
    if scale == 1:
        return image
    else:
        image = np.repeat(image, scale, axis=0)
        image = np.repeat(image, scale, axis=1)
        return image
    
    
def segment_with_mean(image, segmentation):
    mean_image = np.zeros_like(image)
    props = regionprops(segmentation + 1)
    
    for i, prop in enumerate(props):
        min_row, min_col, max_row, max_col = prop['bbox']
        sliced_image = image[min_row:max_row, min_col:max_col]
        color = sliced_image[prop['image']].mean(axis=0)
        
        for row, col in prop['coords']:
                mean_image[row, col] = color
        
    return mean_image
def own_data_augmentation_(cropped_image,cropped_mask,output_image,output_mask,count,image_name):

  #bright =========================================================================================
  bright = np.ones(cropped_image.shape , dtype="uint8") * 70
  brightincrease = cv2.add(cropped_image,bright)
  #Sharpening =================================================================================================
  sharpening = np.array([ [-1,-1,-1],
                          [-1,10,-1],
                          [-1,-1,-1] ])
  sharpened = cv2.filter2D(cropped_image,-1,sharpening)
  cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) +'_sharp_'+ image_name), sharpened) 
  cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) +'_sharp_'+ image_name), cropped_mask)   
  #FLipping ===========================================================================================
  flipHorizontal_image = cv2.flip(cropped_image, 1)
  flipHorizontal_mask = cv2.flip(cropped_mask, 1)
  cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) +'_flip1_'+ image_name), flipHorizontal_image) 
  cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) +'_flip1_'+ image_name), flipHorizontal_mask)   
  
  #resolution change with segmentation ============================================================
  if np.random.randint(2):
    num_sepgment=10000- np.random.randint(5000)
    segments_slic = slic(cropped_image, n_segments=num_sepgment, compactness=5, sigma=0)
    segmentation=segment_with_mean(cropped_image, segments_slic)
    
    cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) +'_slic'+str(num_sepgment)+'c10_'+ image_name), segmentation) 
    cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) +'_slic'+str(num_sepgment)+'c10_'+ image_name), cropped_mask)   

  if np.random.randint(2):
    segments_quick = quickshift(cropped_image, kernel_size=3, max_dist=6, ratio=0.5)
    segmentation=segment_with_mean(cropped_image, segments_quick)
    cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) +'_quickshiftk3m6r05_'+ image_name), segmentation) 
    cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) +'_quickshiftk3m6r05_'+ image_name), cropped_mask)   
  if np.random.randint(2):
    gradient = sobel(rgb2gray(cropped_image))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)  
    segmentation=segment_with_mean(cropped_image, segmentation)
    cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) +'_watershedm250c0001_'+ image_name), segmentation) 
    cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) +'_watershedm250c0001_'+ image_name), cropped_mask)   
  #rotate =============================================================================================
  angle=180 - np.random.randint(170)
  im=rotate_image(cropped_image, angle)
  mask=rotate_image(cropped_mask, angle)
  mask[mask>0]=255

  cv2.imwrite(os.path.join(output_image, 'samples_'+ str(count) +'_angle_'+str(angle) +'_'+ image_name), im) 
  cv2.imwrite(os.path.join(output_mask, 'gt_'+ str(count) +'_angle_'+str(angle) +'_'+ image_name), mask)   


