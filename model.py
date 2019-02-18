import tensorflow as tf
import os 
import math
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
    #specify network
#     x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
#     layer_input_list=[x]
#     with tf.name_scope('linear_model') as scope:
#         for layer in range(0,layers):
#             vars()['hidden_layer_'+str(layer+1)]=tf.layers.dense(layer_input_list[layer], 400, activation=tf.nn.relu,kernel_regularizer=Regularization,bias_regularizer=Regularization, name='hidden_layer_'+str(layer+1))
#             layer_input_list.append(vars()['hidden_layer_'+str(layer+1)])
#         output = tf.layers.dense(vars()['hidden_layer_'+str(layer+1)], 10, name='output_layer')
    
#     tf.identity(output, name='output')    
                    
#     #evaluation
#     y = tf.placeholder(tf.float32, [None, 10], name='label')
    
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
