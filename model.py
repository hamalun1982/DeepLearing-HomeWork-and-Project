import tensorflow as tf
import os 
import math
import numpy as np
import matplotlib.pyplot as plt
def train_function(train_images,train_labels,val_images,val_labels, test_images, test_labels, Regularization, Learn_rate, batch_size, layers):
    """Comment discription here later
    """
    
    # variables
    train_num_examples = train_images.shape[0]
    val_num_examples =  val_images.shape[0]
    test_num_examples = test_images.shape[0]
    
    train_images = np.reshape(train_images, [-1, 32, 32, 3])
    val_images = np.reshape(val_images, [-1, 32, 32, 3])
    test_images = np.reshape(test_images, [-1, 32, 32, 3])

    
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
    filters = [16, 32, 64] 
    layer_input_list=[]
    with tf.name_scope('conv_block') as scope:
       # let's specify a conv stack
        hidden_1 = tf.layers.conv2d(x, 32, 5,
                                    padding='same', 
                                    activation=tf.nn.relu, 
                                    kernel_regularizer=r,
                                    bias_regularizer=r,
                                    activity_regularizer=r,
                                    name='hidden_1')
        pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        hidden_2 = tf.layers.conv2d(pool_1, 64, 5,
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    kernel_regularizer=r,
                                    bias_regularizer=r,
                                    activity_regularizer=r,
                                    name='hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
        print(hidden_2)
        # followed by a dense layer output
        flat = tf.reshape(hidden_2, [-1,8*8*256]) # flatten from 4D to 2D for dense layer
        print(flat)
        output = tf.layers.dense(flat, 100, name='output')
        
#         layer_input_list= [first_conv, second_conv, third_conv, pool]
        
    tf.identity(output, name='output')    
        
#     logits_size=[512,100] labels_size=[128,100] -> output = 512, label = 128
        
    print(output)
    #evaluation
    y = tf.placeholder(tf.float32, [None, 100], name='label')
    print(y)
    
    
    
    
    
    
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(output, 1)),"float"), name='accuracy')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='ce_loss')

    # set up training
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=100)
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = Learn_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
    saver = tf.train.Saver()
    save_directory = './homework1_logs_with_regularization_1'

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        best_val_acc=0
        train_conf_mxs = []
        
        for epoch in range(100):
            print('Epoch',epoch)
            # run gradient steps
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
#                 print(tf.shape(batch_ys))
#                 batch_xs = np.reshape(batch_xs, [-1, 3072])
                _, train_conf_matrix = session.run([train_op, confusion_matrix_op], 
                                                    {x: batch_xs, y: batch_ys})
#                 print('han')
                train_conf_mxs.append(train_conf_matrix)

            print('TRAIN CONFUSION MATRIX:')
            print(str(sum(train_conf_mxs)))
          
            
            #train accuraccy
#             train_acc =  session.run(accuracy, {x:train_images, y: train_labels})
#             print('Training Accuracy',train_acc)
            print('Training Accuracy')
            
            # validation train
            val_ce_vals = []
            val_conf_mxs = []
            for i in range(val_num_examples // batch_size):
                batch_xs = val_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = val_labels[i * batch_size:(i + 1) * batch_size, :]
#                 batch_xs = np.reshape(batch_xs, [-1, 3072])
                val_ce, val_conf_matrix = session.run([tf.reduce_mean(cross_entropy), confusion_matrix_op], 
                                                    {x: batch_xs, y: batch_ys})
                val_conf_mxs.append(val_conf_matrix)

            #val accuracy
            
            
            print('VALIDATION CONFUSION MATRIX:')
            print(str(sum(val_conf_mxs)))
            
            val_acc = session.run(accuracy, {x: val_images, y: val_labels})
            print('val Accuracy',val_acc)
            
              #test accuracy
            
             # test 
            ce_vals = []
            test_conf_mxs = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
#                 batch_xs = np.reshape(batch_xs, [-1, 3072])
                test_ce, test_conf_matrix = session.run([tf.reduce_mean(cross_entropy), confusion_matrix_op], 
                                                    {x: batch_xs, y: batch_ys})
                test_conf_mxs.append(test_conf_matrix)
            print('TEST CONFUSION MATRIX:')
            print(str(sum(test_conf_mxs)))
            
            test_acc = session.run(accuracy, {x: test_images, y: test_labels})
            print('test Accuracy',test_acc)
            
            error =  1 - test_acc
#             const = 0.95 is our confidence interval, so const = 1.96
#             n =  number of the example on the test so it should be test_num_examples 
            confidence_interval_1 = error + 1.96 * math.sqrt((error * (1 - error)) / test_num_examples)
            confidence_interval_2 = error - 1.96 * math.sqrt((error * (1 - error)) / test_num_examples)
             
            print("First Confidence interval is",confidence_interval_1 )
            print("Second Confidence interval is", confidence_interval_2)
            print("\n")
#             early stopping
            if (val_acc > best_val_acc ):
                best_val_acc=val_acc
                best_train_acc=train_acc
                counter=0
            else:
                counter=counter+1
                if counter > 5:
                    break
                else:
                    continue
        print('save model to directory ')
        # this will save the best one among all 
        path_prefix = saver.save(session, os.path.join(save_directory, "homework_1"))
                    

    tf.reset_default_graph()

    return best_train_acc, best_val_acc

def upscale_block(x, scale=2):
    """transpose convolution upscale"""
    return tf.layers.conv2d_transpose(x, 1, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')

def autoencoder_network(x, code_size=100):
    encoder_16 = downscale_block(x)
    print("encoder_16 is", encoder_16)
    encoder_8 = downscale_block(encoder_16)
    print("encoder_8 is", encoder_8)
    encoder_4 = downscale_block(encoder_8)
    print("encoder_4 is", encoder_4)
    flatten_dim = np.prod(encoder_4.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_4, [-1, flatten_dim])
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu)
    hidden_decoder = tf.layers.dense(code, 16, activation=tf.nn.elu)
    decoder_4 = tf.reshape(hidden_decoder, [-1, 4, 4, 1])
    decoder_8 = upscale_block(decoder_4)
    decoder_16 = upscale_block(decoder_8)
    output = upscale_block(decoder_16)
    return code, output


def train_function_with_autoencoder(train_images,train_labels,val_images,val_labels, test_images, test_labels, images_part2, Regularization, Learn_rate, batch_size, layers):
    """Comment discription here later
    """
    
    # variables
    train_num_examples = train_images.shape[0]
    val_num_examples =  val_images.shape[0]
    test_num_examples = test_images.shape[0]
    
    train_images = np.reshape(train_images, [-1, 32, 32, 3])
    val_images = np.reshape(val_images, [-1, 32, 32, 3])
    test_images = np.reshape(test_images, [-1, 32, 32, 3])
    images_part2 = np.reshape(images_part2, [-1, 32, 32, 3])
    #specify network
    
    
    # set hyperparameters
    sparsity_weight = 5e-3
    code_size = 40
    noise_level = 0.1
    
    # define graph
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
    # need to denoise x 
    x_denoise =  x_noisy = x + noise_level * tf.random_normal(tf.shape(x))
    
    code, outputs = autoencoder_network(x_denoise, code_size)

    # calculate loss
    sparsity_loss = tf.norm(code, ord=1, axis=1)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - x)) # Mean Square Error
    total_loss = reconstruction_loss + sparsity_weight * sparsity_loss
    print("total loss is:", total_loss)
    # setup optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss)
    
    #pretraining 
    
    batch_size = 16
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for epoch in range(2):
        for i in range(images_part2.shape[0] // batch_size):
            batch_xs = images_part2[i*batch_size:(i+1)*batch_size, :]
            session.run(train_op, {x: batch_xs})
    print("Finish pretraining!")
    
    #after that connect some connected layers and an output layer
    #set up layer 
    filters = [16, 32, 64] 
    layer_input_list=[]
    with tf.name_scope('conv_block') as scope:
       # let's specify a conv stack
        hidden_1 = tf.layers.conv2d(x, 32, 5,
                                    padding='same', 
                                    activation=tf.nn.relu, 
                                    kernel_regularizer=r,
                                    bias_regularizer=r,
                                    activity_regularizer=r,
                                    name='hidden_1')
        pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        hidden_2 = tf.layers.conv2d(pool_1, 64, 5,
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    kernel_regularizer=r,
                                    bias_regularizer=r,
                                    activity_regularizer=r,
                                    name='hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
        print(hidden_2)
        # followed by a dense layer output
        flat = tf.reshape(hidden_2, [-1,8*8*256]) # flatten from 4D to 2D for dense layer
        print(flat)
        output = tf.layers.dense(flat, 100, name='output')
        
#         layer_input_list= [first_conv, second_conv, third_conv, pool]
        
    tf.identity(output, name='output')    
        
#     logits_size=[512,100] labels_size=[128,100] -> output = 512, label = 128
        
    print(output)
    #evaluation
    y = tf.placeholder(tf.float32, [None, 100], name='label')
    print(y)
    
    
    
    
    
    
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(output, 1)),"float"), name='accuracy')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='ce_loss')

    # set up training
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=100)
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = Learn_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
    saver = tf.train.Saver()
    save_directory = './homework1_logs_with_regularization_1'

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        best_val_acc=0
        train_conf_mxs = []
        
        for epoch in range(100):
            print('Epoch',epoch)
            # run gradient steps
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
#                 print(tf.shape(batch_ys))
#                 batch_xs = np.reshape(batch_xs, [-1, 3072])
                _, train_conf_matrix = session.run([train_op, confusion_matrix_op], 
                                                    {x: batch_xs, y: batch_ys})
#                 print('han')
                train_conf_mxs.append(train_conf_matrix)

            print('TRAIN CONFUSION MATRIX:')
            print(str(sum(train_conf_mxs)))
          
            
            #train accuraccy
#             train_acc =  session.run(accuracy, {x:train_images, y: train_labels})
#             print('Training Accuracy',train_acc)
            print('Training Accuracy')
            
            # validation train
            val_ce_vals = []
            val_conf_mxs = []
            for i in range(val_num_examples // batch_size):
                batch_xs = val_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = val_labels[i * batch_size:(i + 1) * batch_size, :]
#                 batch_xs = np.reshape(batch_xs, [-1, 3072])
                val_ce, val_conf_matrix = session.run([tf.reduce_mean(cross_entropy), confusion_matrix_op], 
                                                    {x: batch_xs, y: batch_ys})
                val_conf_mxs.append(val_conf_matrix)

            #val accuracy
            
            
            print('VALIDATION CONFUSION MATRIX:')
            print(str(sum(val_conf_mxs)))
            
            val_acc = session.run(accuracy, {x: val_images, y: val_labels})
            print('val Accuracy',val_acc)
            
              #test accuracy
            
             # test 
            ce_vals = []
            test_conf_mxs = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
#                 batch_xs = np.reshape(batch_xs, [-1, 3072])
                test_ce, test_conf_matrix = session.run([tf.reduce_mean(cross_entropy), confusion_matrix_op], 
                                                    {x: batch_xs, y: batch_ys})
                test_conf_mxs.append(test_conf_matrix)
            print('TEST CONFUSION MATRIX:')
            print(str(sum(test_conf_mxs)))
            
            test_acc = session.run(accuracy, {x: test_images, y: test_labels})
            print('test Accuracy',test_acc)
            
            error =  1 - test_acc
#             const = 0.95 is our confidence interval, so const = 1.96
#             n =  number of the example on the test so it should be test_num_examples 
            confidence_interval_1 = error + 1.96 * math.sqrt((error * (1 - error)) / test_num_examples)
            confidence_interval_2 = error - 1.96 * math.sqrt((error * (1 - error)) / test_num_examples)
             
            print("First Confidence interval is",confidence_interval_1 )
            print("Second Confidence interval is", confidence_interval_2)
            print("\n")
#             early stopping
            if (val_acc > best_val_acc ):
                best_val_acc=val_acc
                best_train_acc=train_acc
                counter=0
            else:
                counter=counter+1
                if counter > 5:
                    break
                else:
                    continue
        print('save model to directory ')
        # this will save the best one among all 
        path_prefix = saver.save(session, os.path.join(save_directory, "homework_1"))
                    

    tf.reset_default_graph()

    return best_train_acc, best_val_acc