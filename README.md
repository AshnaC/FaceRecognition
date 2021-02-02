# FaceRecognition

The project contains the code for Face Recognition System to identify the last 9 presidents of the US.

A Face recognition system developed using embedding from Google’s FaceNet architecture, trained to detect the last nine presidents of the US.

The implementation is in Python Flask framework with React front end. 

MTCNN Neural network is used to identify the face bounding box of image, which is fed to a pre-learned FaceNet network to get face embeddings.

This embeddings is used to train Support Vector Classifier (SVC) to identify faces.

Direct Deployment to Heroku was not possible due to the large slack size since the Facenet keras model file size exceeds free layer size.

A sample implementation gif is added in the below link for reference.

https://portfolio-apps-ashna.herokuapp.com/face
