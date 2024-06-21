# Star-Detection
Welcome to the Star-Detection project! This cutting-edge machine learning application is designed to analyze live video feeds and accurately identify six-pointed stars in real-time. Whether the stars vary in size, color, or angle, our model is trained to detect them with precision.

With the intention to find any possible errors or area of improvement on the model I added two new scripts:
-visualize_data in order to view a random sample and its label of the data collected.
-test_model in order to test our trained model without running the live feed.

The model works great with a consistent precision score found over the evaluation results.
-The live feed shows a green text over specifying if the figure-label recognized.

I excluded the .keras file for the model and the procesed data.npz as both were too large for github to properly upload without extra .gitattributes.