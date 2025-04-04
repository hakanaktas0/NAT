how to run: python run.py (working dir should be src)

generate_data() function uses 1000 words from the brown dataset, I didnt take the most common 1000 because there were just filler words like to, in etc. so I took between 500 to 1500


some of the code has redundant parts (courtesy of GPT) so if something does not seem to be used, it is not used

both the char embeds and substring embed is 768 dim because of the embedding layer of GPT2

what happens is, the substring embed is multiplied with condition which can be 0 (normal BPE) or 1 (search). then the substring embed is concatanated with the char embed in each node. 

I believe the GNN is not fully connected and only the ones next to each other are connected.

UPDATE : BATCHING IS NOT WORKING FOR SOME REASON, I WILL INVESTIGATE. OTHER THAN THAT, it is very low (sometimes as low as 20 sometimes as high as 80ish) accuracy on validation set before training and over 99 after training. Validation includes sentences formed using the same words in the training set but the samples are different.