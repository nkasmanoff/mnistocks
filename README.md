MNISTocks
=========

A combined approach to predicting the stock market, using trend and sentiment analysis in one model. 


In this project, I explore the applications of convolutional neural networks and 
multi-input functional models as a means to predict whether or not a stock will increase or decrease by a certain amount over a given holding period. 

This can be done for any arbitrary jump/fall of a stock over any given holding period, 
but I will note now that for the experimentation done here, this is performed for identifying stocks that either increase or decrease by 5% over a 10 day holding period, based on that asset's behavior over the previous 10 days, based on that asset's change in adjusted closing price, as well as standard deviation of public sentiment. 


The architecture of the model used is visualized below. 

![My model](https://github.com/nkasmanoff/mnistocks/blob/master/noahmodel.png)


As you can see, a LeNet-esque model, and then attached to it is an additional input before the final dense layer, at which point the classification of either a 5% rise or fall is determined. 

The main input to this model gives motivation to the name of this project, MNISTocks. The input is an image shown below, which is the behavior of the adjusted closing price of a given stock over a 10 day period. As you can see when comparing it to what we know the inputs to the famous MNIST handwritten character dataset. 

Where a positive return may have a profile such as 

![5% return on investment](https://github.com/nkasmanoff/mnistocks/blob/master/AAPLfor10days%255.0on2013-3-9SENTAVG%3D-0.09SENTSTD%3D2.81.png)


And a negative return may look like 

![5% return on investment](https://github.com/nkasmanoff/mnistocks/blob/master/AAPLfor10days%25-5.0on2013-6-10SENTAVG%3D0.73SENTSTD%3D3.5.png)


The motivation for this CNN is that if a very effective model can be created to decipher the differences in characters with an easily trainable network, if such a distinction exists. There is some evidence to suggest there is, seeing as "Trend Analysis" is it's own entry on investopedia, and there does exist an auto-correlation between a stock's price between days. So in an ideal world, there will exist a certain curve which will be found within a convolutional filter, and help distinguish a rising stock from a falling one. We don't live in a perfect world though, and other factors should be considered, such as the sentiment (and proxy of public perception, outpouring of good/bad news) of a stock. 


Sentiment analsysis is already it's own field of stock market prediction. In addition to the signal (hopefully) gained by analyzing a stock's chart with the main input, I also hope to capture the signal left by public news about the company. 

For the sake of simplicity I am assuming this is captured one dataset, (sentex dataset) where a companies sentiment on a given day is recorded as a number between -3 (bad) to 6 (good) I am not sure why this is the scale. By considering factors such as the standard deviation of this number over the holding period (was there a significant news development), and the average (overwhelmingly positive sentiment), my hope is that this encodes more information for this model, further helping capture positive and negative return investments. 


Including this final part into the last Dense layer of the network, such training should hopefully lead to an over 50% classification procderue, and allow for the user of this network to make informed investments. 


I'll quickly disclaim this sentiment data was pulled from the free version, which is only over 2012-2016. In order to use current sentiment values, a paid version or novelly generated one were required, which was not the priority to this point. 



# Code Execution

  To collect data, run python noahtrader.py from your command line, changing the path to wherever you'd like to store these images. 
  
  To train model, run python noahmodel.py from the command line as well, again editting the path, and you'll see how the training performs on a validation set. 
