# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) A Comprehensive Study of the Factors that Affect Home Prices in Ames, Iowa¶

### Problem Statement

As a realtor company operating in Ames, Iowa, our goal is to help our clients make informed decisions when buying or selling a home. One of the key factors that our clients consider is the price of the property, and we want to develop a reliable regression model that can accurately predict the prices of homes in this area. By analyzing a dataset of previous home sales in Ames and identifying the key features that impact home prices, we aim to create a regression model that can be used to predict future prices based on a variety of property characteristics. Our ultimate goal is to provide our clients with a powerful tool that can help them make smart and profitable real estate decisions.

### Background

The real estate market in Ames, Iowa has seen significant growth over the past decade, with a steady increase in home prices and a high demand for quality housing. As a result, there is a growing need for accurate predictions of home prices in this area, both for home buyers looking to make a wise investment and for realtors seeking to offer valuable insights to their clients.

To address this need, our realtor company is undertaking a project to develop a regression model that can predict home prices in Ames based on a variety of key factors. By analyzing a dataset of past home sales and identifying the most significant features that impact home prices, we aim to build a reliable and accurate model that can help our clients make informed decisions about their real estate investments.

This project represents a significant opportunity for our company to provide a valuable service to our clients, while also gaining a deeper understanding of the complex factors that drive home prices in the Ames real estate market. Through careful analysis and rigorous testing, we believe that we can develop a powerful tool that will help our clients maximize their investments and achieve their real estate goals.


---

### Datasets

The raw dataset provided from Kaggle: consisting 80 variables that could impact on house prices.

|Feature|Type|Description|
|---|---|---|
|SalePrice|Numeric(Continious)|the property's sale price in dollars. This is the target variable that you're trying to predict|.
|MSSubClass|Numeric(Discrete)|The building class|
|MSZoning|Object(Cetagorical)|The general zoning classification|
|LotFrontage|Numeric(Continious)|Linear feet of street connected to property|
|LotArea|Numeric(Continious)|Lot size in square feet|
|Street|Object(Cetagorical)|Type of road access|
|Alley|Object(Cetagorical)|Type of alley access|
|LotShape|Object(Cetagorical)|General shape of property|
|LandContour|Object(Categorical)|Flatness of the property|
|Utilities|Object(Categorical)|Type of utilities available|
|LotConfig|Object(Categorical)|Lot configuration|
|LandSlope|Object(Categorical)|Slope of property|
|Neighborhood|Object(Categorical)|Physical locations within Ames city limits|
|Condition1|Object(Categorical)|Proximity to main road or railroad|
|Condition2|Object(Categorical)|Proximity to main road or railroad (if a second is present)|
|BldgType|Object(Categorical)|Type of dwelling|
|HouseStyle|Object(Categorical)|Style of dwelling|
|OverallQual|Numeric(Discrete)|Overall material and finish quality|
|OverallCond|Numeric(Discrete)|Overall condition rating|
|YearBuilt|Numeric(Discrete)|Original construction date|
|YearRemodAdd|Numeric(Discrete)|Remodel date|
|RoofStyle|Object(Categorical)|Type of roof|
|RoofMatl|Object(Categorical)|Roof material|
|Exterior1st|Object(Categorical)|Exterior covering on house|
|Exterior2nd|Object(Categorical)|Exterior covering on house (if more than one material)|
|MasVnrType|Object(Categorical)|Masonry veneer type|
|MasVnrArea|Numeric(Continious)|Masonry veneer area in square feet|
|ExterQual|Object(Categorical)|Exterior material quality|
|ExterCond|Object(Categorical)|Present condition of the material on the exterior|
|Foundation|Object(Categorical)|Type of foundation|
|BsmtQual|Object(Categorical)|Height of the basement|
|BsmtCond|Object(Categorical)|General condition of the basement|
|BsmtExposure|Object(Categorical)|Walkout or garden level basement walls|
|BsmtFinType1|Object(Categorical)|Quality of basement finished area|
|BsmtFinSF1|Numeric(Continious)|Type 1 finished square feet|
|BsmtFinType2|Object(Categorical)|Quality of second finished area (if present)|
|BsmtFinSF2|Numeric(Continious)|Type 2 finished square feet|
|BsmtUnfSF|Numeric(Continious)|Unfinished square feet of basement area|
|TotalBsmtSF|Numeric(Continious)|Total square feet of basement area|
|Heating|Object(Categorical)|Type of heating|
|HeatingQC|Object(Categorical)|Heating quality and condition|
|CentralAir|Object(Categorical)|Central air conditioning|
|Electrical|Object(Categorical)|Electrical system|
|1stFlrSF|Numeric(Continious)|First Floor square feet|
|2ndFlrSF|Numeric(Continious)|Second floor square feet|
|LowQualFinSF|Numeric(Continious)|Low quality finished square feet (all floors)|
|GrLivArea|Numeric(Continious)|Above grade (ground) living area square feet|
|BsmtFullBath|Numeric(Discrete)|Basement full bathrooms|
|BsmtHalfBath|Numeric(Discrete)|Basement half bathrooms|
|FullBath|Numeric(Discrete)|Full bathrooms above grade|
|HalfBath|Numeric(Discrete)|Half baths above grade|
|Bedroom|Numeric(Discrete)|Number of bedrooms above basement level|
|Kitchen|Numeric(Discrete)|Number of kitchens|
|KitchenQual|Object(Categorical)|Kitchen quality|
|TotRmsAbvGrd|Numeric(Discrete)|Total rooms above grade (does not include bathrooms)|
|Functional|Object(Categorical)|Home functionality rating|
|Fireplaces|Numeric(Discrete)|Number of fireplaces|
|FireplaceQu|Object(Categorical)|Fireplace quality|
|GarageType|Object(Categorical)|Garage location|
|GarageYrBlt|Numeric(Discrete)|Year garage was built|
|GarageFinish|Object(Categorical)|Interior finish of the garage|
|GarageCars|Numeric(Discrete)|Size of garage in car capacity|
|GarageArea|Numeric(Continious)|Size of garage in square feet|
|GarageQual|Object(Categorical)|Garage quality|
|GarageCond|Object(Categorical)|Garage condition|
|PavedDrive|Object(Categorical)|Paved driveway|
|WoodDeckSF|Numeric(Continious)|Wood deck area in square feet|
|OpenPorchSF|Numeric(Continious)|Open porch area in square feet|
|EnclosedPorch|Numeric(Continious)|Enclosed porch area in square feet|
|3SsnPorch|Numeric(Continious)|Three season porch area in square feet|
|ScreenPorch|Numeric(Continious)|Screen porch area in square feet|
|PoolArea|Numeric(Continious)|Pool area in square feet|
|PoolQC|Object(Categorical)|Pool quality|
|Fence|Object(Categorical)|Fence quality|
|MiscFeature|Object(Categorical)|Miscellaneous feature not covered in other categories|
|MiscVal|Numeric(Continious)|$Value of miscellaneous feature|
|MoSold|Numeric(Discrete)|Month Sold|
|YrSold|Numeric(Discrete)|Year Sold|
|SaleType|Object(Categorical)|Type of sale|
|SaleCondition|: |Condition of sale|

---

### Exploratory Data Analysis and Pre-Processing (EDA)

The exploratory data analysis (EDA) and preprocessing steps involved in this project focused on cleaning and preparing the data for modeling. The EDA phase involved analyzing the data and identifying patterns, trends, and relationships between variables. This included examining the target sale price and its predictors, both numerical and categorical. The preprocessing phase involved several steps, such as handling missing values by dropping columns with more than 10% null values, utilizing imputation techniques, and removing some null values. Outliers were identified and addressed using scatter plots, and categorical variables were transformed using value mapping and one-hot encoding. The neighborhood names were also changed to be more understandable.

### EDA

we could look into the main steps that we made through Exploratory Data Analysis:

 **1. Analyzed the target variable (sale price) by examining its distribution** 


![house_sale_price_distribution](images/house_sale_price_distribution.png)

**Number of houses with more than 500,000 sale price: 12** 
**Further in pre-processing we decided to temoved some sale prices that were way outside of the majority of (prices more than 500,000)**


**2. Check the errors (incompatibility) in relationship between basement and Ground variables**
For some variables, such as those related to basement and above ground levels, we examined their compatibility to ensure they were consistent with each other. While there were no issues with the basement variables, we did find some incompatibility in the ground level variables, which we identified and addressed.

**Incompatibility in Ground levels**

![sample_image_captions](images/sample_image_captions.png)

We Utilized imputation techniques to address incompatibilities in ground levels

### Preprocessing
The overall aim of the Data Preprocessing phase is to clean and transform the data to make it more suitable for modeling, handle missing data, reduce noise, and improve the performance of the model. This phase typically involves several steps as following:

**Preprocessing Steps**

**1. Handling missing values**
We assessed missing values across all variables including numerical and categorical ones and utilized different techniques based on the nature of the data and the amount of missingness. In the following we could see numerical variables with their missing values:

![missing_values](images/missing_values.png)

- After identifying missing values in numerical variables, we handled them by dropping columns with more than 10% null values, utilizing imputation techniques, and dropping some null values. 
- We also utilized imputation technique for filling missing values in categorical variables.

**2. 


- Remove special characters and numbers present in the text
- Remove punctuations in the text
- Remove extra spaces
- Remove single characters
- Add starting and ending tags to the sentences to indicate the beginning and the ending of a sentence (By adding start and end tags, we are essentially giving the model a cue as to when the caption starts and ends. During training, the model learns to predict the next word in the caption given the previous words. By adding the start tag at the beginning, the model knows that it needs to start generating a caption, and by adding the end tag at the end, the model knows that it has completed generating the caption.)

### Exploratory Data Analysis (EDA)
In image captioning, the length of the generated captions can have a significant impact on the performance and quality of the model. It is therefore important to understand the distribution of caption lengths in the training data. One way to visualize this distribution is to use a histogram.

The resulting histogram can provide insights into the distribution of caption lengths in the training data. For example, it may reveal that most captions are relatively short, with a few longer captions that are outliers. This information can be used to inform the design and development of the image captioning model, such as adjusting the maximum caption length or exploring techniques for handling longer captions.
The below histogram shows the distribution of captions lengths. It shows that the most captions length are 10 and it has the normal distribution. By this histogram we will set the maximum caption length to 25. By setting a maximum caption length, we ensure that the model can handle captions of different lengths, making it more robust and generalizable. If a caption in the training set is longer than 15 words, it will be truncated to 15 words. Similarly, if a caption is shorter than 15 words, it will be padded with special tokens to reach the desired length.

![Distribution_Words_In_Caption](images/Distribution_Words_In_Caption.png)

#### Vectorize the text
The resulting plot can provide insights into the most frequent words. By having this plot we could see that most images are about people and dogs, explaining what they are wearing or doing with color detail descriptions.

![Words_Most_Frequencies](images/Words_Most_Frequencies.png)


### EDA and Cleaning Summary

- We analyzed the image and caption data to get an understanding of the dataset's characteristics.
- Cleaned the captions by applying several pre-processing steps, such as lowercasing, removing special characters, extra spaces, and single characters. Beside that, we added start and end tokens. we are essentially giving the model a cue as to when the caption starts and ends.
- Created a new dataset that includes cleaned image-caption pairs. This cleaned dataset will serve as the input for training the image caption model.
- Also plotted a histogram of caption lengths to understand the distribution of caption lengths in the cleaned dataset. This visualization helps inform the choice of hyperparameters for the model, such as the maximum caption length.

---

## Image Feature Extraction
Image feature extraction is a critical step in image captioning, as it involves converting raw image data into a set of numerical features that can be used as input to the image captioning model. One way to perform to do is to use a pretrained deep learning CNN models such as VGG16, ResNet, Densenet201 or Inception. These models are trained on large datasets such as ImageNet (a large-scale dataset consisting of more than 14 million labeled images across 20,000 different categories), and have learned to extract meaningful and discriminative features from images.

The process of using a pretrained model for image feature extraction involves the following steps:

- Load the Pretrained Model
- Remove the Classification Layers: These layers are designed to predict the class label of an image, which is not relevant for image feature extraction.
- Extracting Image Features: This is done by passing each image through the modified model, and obtaining the output of one of the intermediate layers. The output of this layer represents a set of high-level features that capture the visual content of the image.
- Save the Extracted Features

Overall, using a pretrained CNN model for image feature extraction is a powerful technique that can significantly improve the performance of image captioning models. By leveraging the power of these models, we can obtain a rich representation of the visual content of images, which can be used to generate accurate and informative captions.

### CNN Pre-Trained Models
Here we will utilize Densenet201 and VGG16 for extracting features from images. VGG16 is a relatively simple architecture compared to DenseNet201, with 16 layers of convolution and pooling operations. It has been widely used for transfer learning in image classification tasks and has achieved state-of-the-art performance on several benchmark datasets. However, VGG16 is computationally expensive and can be slow to train.

DenseNet201, on the other hand, has more layers and a more complex architecture that allows for better feature reuse and can improve the flow of information through the network. It has also shown good performance on a variety of image classification tasks, and is relatively computationally efficient compared to some other deep neural network architectures.

In general, if you have a large dataset and computational resources, DenseNet201 may be a better choice, as it can capture more complex features and can potentially achieve better performance. If you have limited data and computational resources, VGG16 may be a better choice, as it is simpler and faster to train.

### Pre-Trained DenseNet201 Model
We set the input layer to be the same as the input layer of the original DenseNet201 model, and the output layer to be the second-to-last layer of the original model. This removes the final classification layer, which was responsible for predicting the class labels of the input images. The below image shows us the architecture of Densenet201 which is dowloaded from https://www.researchgate.net/figure/DenseNet201-architecture-with-extra-layers-added-at-the-end-for-fine-tuning-on-UCF-101_fig2_353225711 .


![DenseNet201_architecture](images/DenseNet201_architecture.png)
**DenseNet201 architecture with extra layers added at the end for fine tuning**


### Pre-Trained VGG16 Model
Like DenseNet 201, we set the input layer to be the same as the input layer of the original DenseNet201 model, and the output layer to be the second-to-last layer of the original model. This removes the final classification layer, which was responsible for predicting the class labels of the input images. The below image shows us the architecture of VGG16 which is dowloaded from https://www.hindawi.com/journals/am/2023/6147422/ .

![VGG16_network_architecture](images/VGG16_network_architecture.png)

### Image Feature Extraction Summary
- loaded two pre-trained models, VGG16 and DenseNet201.
- Removed the classification layer from both models.
- Passed each image in our dataset through each of these models to obtain a set of features for the image.
- Extracted features stored in separate files.

**These extracted features will be used as input to train an image captioning model**

---

## Image Captioning Models
The goal of this project as explained before, is to create an image captioning model that can generate descriptive and accurate captions for images. To achieve this, we first explored the dataset and cleaned the captions, by visualizing the distribution of caption lengths, we determined the maximum caption length. We then utilized pre-trained CNN models, VGG16 and DenseNet201, to extract image features which will be passed through an LSTM model to generate captions. To enhance the quality of the captions, we will also utilize GloVe embeddings to represent the words in the captions, allowing the model to better understand the semantic meaning of the words. During training, the models will be trained using a large dataset of image-caption pairs, and the weights of the model will be updated using backpropagation to minimize the loss function. Once trained, the model can be used to generate captions for new images by passing the image through the CNN and then feeding the extracted features into the LSTM to generate the caption word by word. Overall, the CNN-LSTM combination is a powerful approach for image captioning that leverages the strengths of both architectures, and has been shown to achieve state-of-the-art results on various benchmarks.

![CNN_LSTM_Model](images/CNN_LSTM_Model.png)

### Data Generation
- This model is exactly the same as the previous CNN with regularization techniques implemented, with an addition of a data augmentation technique.
- Because the audio data has essentially been transcoded into something similar to an image, flipping the "image" can effectively add more diverse data for the model to train on. This flipping is analogous to feeding the audio clip through the model in reverse.
- Adding this diversity in data can improve the model without having to truly provide it more data.
- As can be seen below, the model begins overfitting very early on, similar to the other CNNs. It also continues improving like the CNN with regularization. The difference is that it learns a little slower, but keeps a closer range between the train and test data, allowing for an increase in overall performance.

### Tokenization and Model with GloVe's Pretrained Embeddings
The words in a sentence are separated/tokenized and encoded in a one hot representation. These encodings are then passed to the embeddings layer to generate word embeddings. The basic idea behind the GloVe word embedding is to derive the relationship between the words from statistics. Unlike the occurrence matrix, the co-occurrence matrix tells you how often a particular word pair occurs together. Each value in the co-occurrence matrix represents a pair of words occurring together.

### Modelling With Extracted Image Features From DenseNet201 and VGG16
The image embedding representations are concatenated with the first word of sentence ie. starsen and passed to the LSTM network and the LSTM network starts generating words after each input thus forming a sentence at the end.
We used The same model architecture for both extracted image features from DenseNet201 and VGG16 to have a clear evaluation by having the same conditions. For both models we leveraged normalization techniques like dropout to prevent overfitting, beside that we defined callbacks like ModelCheckpoint, EarlyStopping and ReduceLROnPlateau for monitoring and reducing the loss score.

**Image Caption Model Architecture**

![Image_Caption_Model](images/Image_Caption_Model.png)


In the following we could see the plots showing the loss and accuracy score over the number of epochs in each model.


**Accuracy and Loss Score With DensNet201-LSTM Model**

![Accuracy_Loss_DensNet201_LSTM_Model](images/Accuracy_Loss_DensNet201_LSTM_Model.png)


**Accuracy and Loss Score With VGG16-LSTM Model**

![Accuracy_Loss_VGG16_LSTM_Model](images/Accuracy_Loss_VGG16_LSTM_Model.png)


|**Model**|**Accuracy**|**Loss**|
|---|---|---|
|**DensNet201-LSTM**|0.30|3.74
|**VGG16-LSTM**|0.31|3.69

In the following we could see some result samples of images with their generated captions by our model.


![Sample_Results_of_Generated_Captions](images/Sample_Results_of_Generated_Captions.png)


---

## Text-To-Speech System For Generated Captions
Text-to-speech is the process of converting written text into spoken words. This technology has numerous applications, including assisting people with visual impairments to access digital content such as web pages, books, and other forms of text-based media.

In the context of our project, text-to-speech can be used to generate audio descriptions for the images that we have captioned. This can be particularly useful for blind or visually impaired individuals who may not be able to see the images themselves, but can still benefit from a verbal description of the content.

To generate audio descriptions for the images in our project, we used gTTS to convert the image captions into spoken words.

---

### Conclusions and Recommendations

### Conclusions
- Based on the project, we have successfully developed an image captioning model that can generate captions for images. The model combines both Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks to generate captions that are relevant to the image content. We have also utilized GloVe word embeddings to improve the accuracy of the generated captions.
- To extract the image features, we have used pre-trained models, VGG16 and DenseNet201, and found that final model (CNN-LSTM) performance with VGG16 is slightly better in terms of loss and accuracy score.
- Finally, we have utilized text-to-speech technology to convert the generated captions into audio descriptions for visually impaired individuals.

### Recommendations
- We suggest further exploration can be conducted on the impact of different pre-trained models and different word embeddings on the accuracy of the model.
- Additionally, using larger datasets, such as Flickr30k, MSCOCO and SBU can potentially improve the accuracy of the image captioning model as it provides more diverse and comprehensive data for the model to learn from. In addition, working with larger datasets can help to reduce overfitting and increase the generalization ability of the model.
- Moreover, we should consider expanding the application of the model. For example, we could use the same model to generate descriptions for videos or to generate captions for images in a different domain, such as medical imaging.

Overall, this project showcases the potential of using machine learning techniques to develop solutions that can enhance accessibility and inclusivity for people with disabilities.
