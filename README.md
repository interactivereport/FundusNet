# FundusNet
FundusNet: a deep learning approach for identifying novel endophenotypes for neurodegenerative and eye diseases from fundus images

![image](https://github.com/user-attachments/assets/3c3d27d1-bcca-4a54-a627-4cb654eb5b26)


Hu, W., Li, K., Gagnon, J., Wang, Y., Raney, T., Chen, J., Chen, Y., Okunuki, Y., Chen, W., & Zhang, B. (2025). FundusNet: A Deep-Learning Approach for Fast Diagnosis of Neurodegenerative and Eye Diseases Using Fundus Images. Bioengineering, 12(1), 57. https://doi.org/10.3390/bioengineering12010057

## Steps:
1. git clone the repo
2. Execute either shgender.sh or shage.sh to run individual CNN or ViT models:\
   a. This process will split the image dataset into training and testing sets, train the CNN/ViT models on the training data, and evaluate them on the test data.\
   b. Users must provide the following inputs:\
    'name of csv_file (string)': Path to the CSV file containing annotations.\
    'root_dir (string)': Directory containing all images.
3. Combine the results using majority voting for ensemble prediction.
