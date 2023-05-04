import matplotlib.pyplot as plt

#I = plt.imread('./data_vid/data/video%04d.jpg' % image_number)
I=plt.imread('./data/video-bilder/image34.jpg')
plt.imshow(I)
plt.show()
#plt.scatter(*u[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=60, label='Observed')
#plt.scatter(*self.hat_u, color='red', label='Predicted', s=10)
#plt.legend()
#plt.title('Reprojected frames and points on image number %d' % image_number)
#draw_frame(self.K, self.transform_camera_base_point, scale=0.025)
#draw_frame(self.K, self.wrist_to_camera, scale=0.025)
#draw_frame(self.K, self.index_first_to_camera, scale=0.025)
#draw_frame(self.K, self.index_second_to_camera, scale=0.025)
#draw_frame(self.K, self.index_third_to_camera, scale=0.025)
#draw_frame(self.K, self.index_fingertip_to_camera, scale=0.025)

#plt.xlim([0, I.shape[1]])
#plt.ylim([I.shape[0], 0])