def train_functions(train_images,train_labels,val_images,val_labels, Regularization, Learn_rate, batach_size, layers):
    """Comment discription here later

    """
    
    #set Regularization as default if none 
    if Regularization == None:
        Regularization = tf.contrib.layers.l2_regularizer(scale=0.01)
    
    # set up results
    columns = ["Layers","Accuracy"]
    for i in range (0, layers):
        val =  "Layer_" + str(i+1) + "_Nodes"
        columns.append(val)
    results=pd.DataFrame(columns)
    
    train_num_examples = train_images.shape[0]
    val_img_examples =  val_images.shape[0]
    
    
    hidden_nodes  = []
    hidden_list = []
    for i in range (0, layers):
        hidden_nodes.append(randint(nodes_min,nodes_max))
      #specify network
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    hidden_list.append(x)
        with tf.name_scope('linear_model') as scope:
            for i in range (0,layers)
                hidden_list[i+1] = tf.layers.dense(hidden_list[i], hidden_nodes[i],
                                         kernel_regularizer=Regularization,
                                         bias_regularizer=Regularization,
                                         activation=tf.nn.relu, name='hidden_layer')
            output = tf.layers.dense(hidden_list[layers], 10, 
                                     kernel_regularizer=Regularization,
                                     bias_regularizer=Regularization,
                                     name='output_layer')
            tf.identity(output, name='output')
    
    #evaluation
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    accuracy=tf.metrics.accuracy(y,predictions,name='accuracy')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='ce_loss')

    # set up training
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = Learn_rate)
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
                    
                    #validation accuracy
                    val_acc = session.run(accuracy, {x: val_images,y: val_labels})
                    #train accuraccy 
                    train_acc =  session.run(accuracy, {x: train_images, y: train_labels})
                    #early stopping
                    if (val_acc < best_val_acc ):
                        best_val_acc=val_acc
                        counter=0
                    else:
                        counter=counter+1
                    if counter > 5:
                        results.iloc[run,0]=num_hlayers
                        results.iloc[run,1]=best_val_acc
                        for i in range (0, layers):
                            results.iloc[run,i+2] = hidden_nodes[i]
                        break
                    else:
                        continue
                    
#     results.to_csv(outfilepath+'arc_search_results.csv',index=False)
#     print("Run",run,"Complete")
#     tf.reset_default_graph()
    return results
