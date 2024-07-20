PERCEPTION ASSIGNMENT REPORT:



TASK 1:self built neural network (model.ipynb)

1 Started off by building a straight forward neural network which had average performance .Hand tuned the parameters for some while to get about a loss(mean squared) fluctauting a lot from 0.5 and upto 4.The average time taken by this model was also pretty high (about 2-3 ms).<br>
2 Adding dropout and BatchRegularization helped in improving accuracy and also added an additional feature layer defined as the "ratio of height and width" which objectivelly imporoved the redundancy of the basic model(manual inference).<br>
3 Finally executed hyperparameter tuning in a separate file for the model(hyperparamter_tuning.ipynb) and found the optimal values for number of neurons in the layer , dropout value ,learning rate and activation function.I optimized the value of number of layers manually (since had a choice only between 1 and 2) by trying the best hyperparameter values of both in model.ipynb and single layer was better in both accuracy and latency. Also manually checked whether early stopping was required(early stopping reduced the accuracy significntly and hence was not executed -avg = 1.2 vs 0.33 over same splits).Instead added learning rate schedular  some epochs after the point where early stopping was taking place for a slight but visible improvement in accuracy.(Tuning data is stored in my_dir)<br>
4 Currently the loss function was consistently <0.5 and the time taken by the model was under 1ms (not great but better)<br>
5 Next, to reduce latency ,I implemented post training quantization in the model (data in post_training_quantized_model.tflite) .Interpreted and implemented the model in tflite.py .This reduced the time taken by a factor of 10^-3 and it was now in micro seconds(1-2microseconds).The mean squared loss was stll under 1.5 and more importantly it was consistent and not fluctauting.Though the loss is higher(<1.5 vs <0.5) , the qunatized model is a much more practical aproach.<br>



TASK 2:Previously built model and fine tuning according to the my requirement(external_model.ipynb)

(Finding a decent use case model online took a lllot of time )
Research paper I came across : https://www.mdpi.com/2073-8994/14/12/2657
This project had a part where using bounding boxes and a lot of other data ,'zloc' was estimated.For this they compared using eXtreme Gradient Boosting (XGBoost) , Random Forest (RF) , and Long Short-Term Memory (LSTM) for predicting the distances.Even though the project inferred LTSM as the best model , i used random forests because the data set was a lot smaller and cleaner for my use case and hence a simpler model was preffered especially keeping latency in mind.<br>
Github link for the project: https://github.com/KyujinHan/Object-Depth-detection-based-hybrid-Distance-estimator/tree/master
Github link for the code of random trees i used:https://github.com/KyujinHan/Object-Depth-detection-based-hybrid-Distance-estimator/blob/master/odd_train/RandomForest_train_sample.ipynb<br>
1 I first changed the data/parameters which were used for the model and already the performance was pretty decent(time per computation:order of 0.1ms loss(rmse) = 0.3)<br>
2 The project also implemented manual hyperparameter tuning which i adjusted for my demands. I wanted to address the latency along with the accuracy and hence used my own functions for assessing the model while tuning.<br>
3 Noticing the common values (of loss and time taken)which came up in iterations I set the function as (para = avg_time*1000 + loss) and found the values of hyperparamters for which this is the minimum.This finally gave me a loss function <0.3 and time in the order of 10^-5.Just for confirming i changed the evaluating function to (para = avg_time*100000 + loss) and the parameters obtained where almost the same.<br>
4 Hence this was probably the best model till now . Even though this is not the best model out of the 3 in latency , it is the best as an overall balance between the latency and redundancy.<br>

SUMMARY:
|                |    Basic_model  |    Quantized_model    |   External_model |
|----------------|-----------------|-----------------------|------------------|
|loss(mse)       |     <0.5        |    <2                 |   <0.3           |
|time/prediction |  1millisecond   | <2 microseconds       |<20 microseconds  |

