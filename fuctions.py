def split_data(data, train_frac, val_frac, test_frac, seed):

    """

    Split a numpy array into three parts for training, validation, and testing.



    Args:

        - data: numpy array, to be split along the first axis

        - train_frac, fraction of data to be used for training

        - val_frac, fraction of data to be used for validation

        - test_frac, fraction of data to be used for testing

        - seed, random seed for reproducibility
    Returns:

        - Training Set

        - Validation Set

        - Testing Set
    
    
    """
    if ((train_frac+val_frac+test_frac) !=1):
        print("ERROR: Train, validation, and test fractions must sum to one.")
    else:
            
        np.random.seed(seed)

        size = data.shape[0]

        split_train = int(train_frac * size)

        split_val = int(val_frac * size)+split_train
       
        np.random.shuffle(data)

        return data[:split_train], data[split_train:split_val], data[split_val:]


def architecture_search(train_images,train_labels,val_images,val_labels,outfilepath='$WORK',hlayers_max=3,nodes_min=10,nodes_max=784)
'''Function conducts a random search for NN architecture. Supports search for between 1 and 3 hidden layers.

-train images: array of train images
-train labels: array of train labels as one hot vectors
-val images: array of validatio images
-val labels: array of validations labels as one hot vectors
-outfilepath: filepath to write results
-hlayers_max: maximum number of hidden layers
-nodes_min: minimum number of nodes in a given layer
-nodes_max: maximum number of nodes in a given layer


'''
    #number of hidden layers
    results=pd.DataFrame(columns=["Layers","Layer_1_Nodes","Layer_2_Nodes","Layer_3_Nodes","Accuracy"])
    train_num_examples = train_images.shape[0]
    batch_size=100
    for run in range(25):
        num_hlayers=randint(1,hlayers_max)
        if (num_hlayers == 1):
            h1_nodes=randint(nodes_min,nodes_max)

            #specify network
            x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
            with tf.name_scope('linear_model') as scope:
                hidden = tf.layers.dense(x, h1_nodes, activation=tf.nn.relu, name='hidden_layer')
                output = tf.layers.dense(hidden, 10, name='output_layer')
                tf.identity(output, name='output')

            # set up training
            global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)

            #evaluation
            y = tf.placeholder(tf.float32, [None, 10], name='label')
            accuracy=tf.metrics.accuracy(y,predictions,name='accuracy')
            mean_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='mean_ce_loss'))


            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                # run training
                for epoch in range(100):

                    # run gradient steps 
                    for i in range(train_num_examples // batch_size):
                        batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                        batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
                        session.run([train_op, mean_cross_entropy)], {x: batch_xs, y: batch_ys})

                    #validation
                    val_acc = session.run(accuracy, {x: val_images,y: val_labels})

                    #early stopping
                    if (val_acc < best_val_acc ):
                        best_val_acc=val_acc
                        counter=0
                    else:
                        counter=counter+1
                    if counter > 5:
                        results.iloc[run,0]=num_hlayers
                        results.iloc[run,1]=h1_nodes
                        results.iloc[run,2]='NA'
                        results.iloc[run,3]='NA'
                        results.iloc[run,4]=best_val_acc
                        break
                    else:
                        continue

        elif (num_hlayers == 2):
            h1_nodes=randint(nodes_min,nodes_max)
            h2_nodes=randint(nodes_min,nodes_max)
            
            #specify network
            x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
            with tf.name_scope('linear_model') as scope:
                hidden1 = tf.layers.dense(x, h1_nodes, activation=tf.nn.relu, name='hidden_layer1')
                hidden2 = tf.layers.dense(hidden1, h2_nodes, activation=tf.nn.relu, name='hidden_layer2')
                output = tf.layers.dense(hidden2, 10, name='output_layer')
                tf.identity(output, name='output')


            #evaluation
            y = tf.placeholder(tf.float32, [None, 10], name='label')
            accuracy=tf.metrics.accuracy(y,predictions,name='accuracy')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='ce_loss')

            # set up training
            global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                # run training
                for epoch in range(100):

                    # run gradient steps 
                    for i in range(train_num_examples // batch_size):
                        batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                        batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
                        session.run([train_op, tf.reduce_mean(cross_entropy)], {x: batch_xs, y: batch_ys})

                    #validation
                    val_acc = session.run(accuracy, {x: val_images,y: val_labels})

                    #early stopping
                    if (val_acc < best_val_acc ):
                        best_val_acc=val_acc
                        counter=0
                    else:
                        counter=counter+1
                    if counter > 5:
                        results.iloc[run,0]=num_hlayers
                        results.iloc[run,1]=h1_nodes
                        results.iloc[run,2]=h2_nodes
                        results.iloc[run,3]='NA'
                        results.iloc[run,4]=best_val_acc
                        break
                    else:
                        continue


        elif num_hlayers == 3:
            h1_nodes=randint(nodes_min,nodes_max)
            h2_nodes=randint(nodes_min,nodes_max)
            h3_nodes=randint(nodes_min,nodes_max)

            #specify network
            x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
            with tf.name_scope('linear_model') as scope:
                hidden1 = tf.layers.dense(x, h1_nodes, activation=tf.nn.relu, name='hidden_layer1')
                hidden2 = tf.layers.dense(hidden1, h2_nodes, activation=tf.nn.relu, name='hidden_layer2')
                hidden3 = tf.layers.dense(hidden2, h2_nodes, activation=tf.nn.relu, name='hidden_layer2')
                output = tf.layers.dense(hidden3, 10, name='output_layer')
                tf.identity(output, name='output')

            # set up training
            global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)

            #evaluation
            y = tf.placeholder(tf.float32, [None, 10], name='label')
            accuracy=tf.metrics.accuracy(y,predictions,name='accuracy')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='ce_loss')


            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                # run training
                for epoch in range(100):

                    # run gradient steps 
                    for i in range(train_num_examples // batch_size):
                        batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                        batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
                        session.run([train_op, tf.reduce_mean(cross_entropy)], {x: batch_xs, y: batch_ys})

                    #validation
                    val_acc = session.run(tf.argmax(val_labels,1),tf.argmax(output,1), {x: val_images,y: val_labels})

                    #early stopping
                    if (val_acc < best_val_acc ):
                        best_val_acc=val_acc
                        counter=0
                    else:
                        counter=counter+1
                    if counter > 5:
                        results.iloc[run,0]=num_hlayers
                        results.iloc[run,1]=h1_nodes
                        results.iloc[run,2]=h2_nodes
                        results.iloc[run,3]=h3_nodes
                        results.iloc[run,4]=best_val_acc
                        break
                    else:
                        continue


        results.to_csv(outfilepath+'arc_search_results.csv',index=False)
        print("Run",run,"Complete")
        tf.reset_default_graph()
