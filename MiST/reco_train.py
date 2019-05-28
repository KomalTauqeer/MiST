# imports from standard python
from __future__ import print_function
# import os
# import sys

# imports from local packages
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
# imports from pip packages

# import six

# imports from MiST
from MiST import data
from MiST import utilis
from MiST import globaldef as gl
from MiST import matching


def init(arg):

    # not needed, but makes it nicer
    opath = gl.arg['mva_path'] + '/'

    utilis.training_path(opath)

    # write hash
    utilis.write_hash()

    do_plot_inputvars = gl.arg['plotvars']
    # override for testing
    do_plot_inputvars = True
    debug = gl.arg['debug']

    # prepare the data
    print('Loading data...')

    reco_data = data.DataHandlerTrainReco()
    reco_data.add(gl.arg['reco_samples'], gl.arg['tree'])
    reco_data.print_all()

    print(reco_data.dataobj)

    match = matching.MCTruthMatching()
    match.add(reco_data)
    match.apply_matching('thq')
    data_train, data_test = match.data_train_test()

    print(data_train)
    print(data_test)


    '''
    data_train, label_train = data.binary_labels(train_sig.data(), train_bkg.data())
    data_test, label_test = data.binary_labels(test_sig.data(), test_bkg.data())

    vars_train = data.get_variables(data_train)
    vars_test = data.get_variables(data_test)

    weights_train = data.get_weights(data_train)
    weights_test = data.get_weights(data_test)

    print(vars_train)
    print()
    print(weights_train)
    print()
    print(vars_train[0])
    print()
    print(vars_train[1])
    print()
    print(vars_train[0:,0])
    print()
    print(vars_train[0:,1])
    print()
    print(vars_train[0:,2])
    print()
    '''

'''
    # TODO: implement correctly
    do_plot_model=True

    # not needed, but makes it nicer
    opath = arg['mva_path'] + '/'

    utilis.training_path(opath)

    batch_size = arg['batchsize']

    arg['batchsize'] = 4096
    epochs = arg['epochs']
    training_fraction = arg['trainingsize']

    do_plot_inputvars=arg['plotvars']
    # override for testing
    #do_plot_inputvars=True
    debug=arg['debug']

    df_sig = rootIO.root2df(arg['signal'], arg['tree'])
    df_bkg = rootIO.root2df(arg['background'], arg['tree'])

    vlist = ['j_pt',
             'j_e',
             'j_phi',
             'j_eta',
             'j_csv',
             'lep_pt',
             'lep_e',
             'lep_eta',
             'lep_phi',
             'met_pt',
	     'wolframh0_30',
	     'rec_delta_phi_top_from_leadingbjet_subleadingbjet',
	     'rec_delta_phi_top_from_subleadingbjet_leadingbjet',
	     'mtw',
             'rec_top_m',
	     'rec_delta_R_top_bjet_from_w',
	     'rec_diff_top_bjet_from_w_pt',
	     'rec_delta_phi_top_leadingbjet',
	     'wolframh3_30',
	     'rec_delta_eta_top_from_subleadingbjet_leadingbjet',
	     'rec_delta_eta_top_bjet',
	     'rec_delta_phi_bjet_bjet',
	     'rec_lep_pt',
	     'rec_delta_eta_lep_leadingbjet',
	     'sphericity_30',
	     'aplanarity_30',
             'sumHtTotal_tag']


    df_full = pd.concat((df_sig, df_bkg), ignore_index=True)


    y = []
    for _df_full, ID in [(df_bkg, 0), (df_sig, 1)]:
            y.extend([ID] * _df_full.shape[0])
    y = np.array(y)


    #df_sig =  df_full_sig[vlist]
    #df_bkg =  df_full_bkg[vlist]

    #print(df_sig)
    #print(df_bkg)

    vlist_j = [key for key in vlist if key.startswith('j_')]
    vlist_l = [key for key in vlist if key.startswith('lep_')]
    vlist_hl = [key for key in vlist if not key.startswith(('j_', 'lep_')) ]

    #print(vlist_hl)
    #print(vlist_j)

    df_j = df_full[vlist_j].copy()
    df_l = df_full[vlist_l].copy()
    df_hl = df_full[vlist_hl].copy()

    #print(df_j)

    n_j = max([len(j) for j in df_j.j_pt])
    n_l = max([len(j) for j in df_l.lep_pt])

    # get index
    ix = range(df_full.as_matrix().shape[0])
    #print(ix)
    y_train, y_test, ix_train, ix_test = train_test_split(y, ix, test_size=1-arg['trainingsize'], train_size=arg['trainingsize'])

    X_j_train, X_j_test = datahandler.create_stream(df_j, n_j, ix_train, ix_test, sort_col='j_pt')
    X_l_train, X_l_test = datahandler.create_stream(df_l, n_l, ix_train, ix_test, sort_col='lep_pt')
    X_hl = df_hl.as_matrix()
    X_hl_train, X_hl_test = X_hl[ix_train], X_hl[ix_test]

    datahandler.define_transform(X_hl_train, opath)
    X_hl_train = datahandler.apply_transform(X_hl_train, opath)
    X_hl_test = datahandler.apply_transform(X_hl_test, opath)


    #jet_channel = Sequential()
    #lep_channel = Sequential()

    JET_SHAPE = X_j_train.shape[1:]
    LEP_SHAPE = X_l_train.shape[1:]
    HL_SHAPE = X_hl_train.shape[1:]


    jet_channel_input = Input(shape=JET_SHAPE, name='jet_input')
    jet_channel_seq = Masking(mask_value=-999, name='jet_masking')(jet_channel_input)
    jet_channel_seq = GRU(25, name='jet_gru')(jet_channel_seq)
    jet_channel_seq = Dropout(0.3, name='jet_dropout')(jet_channel_seq)

    lep_channel_input = Input(shape=LEP_SHAPE, name='lep_input')
    lep_channel_seq = Masking(mask_value=-999, name='lep_masking')(lep_channel_input)
    lep_channel_seq = GRU(25, name='lep_gru')(lep_channel_seq)
    lep_channel_seq = Dropout(0.3, name='lep_dropout')(lep_channel_seq)

    hl_channel_input = Input(shape=HL_SHAPE, name='hl_input')
    hl_channel_seq = Dense(100, activation='relu')(hl_channel_input)
    hl_channel_seq = Dropout(.25)(hl_channel_seq)

    merge_seq = concatenate([jet_channel_seq, lep_channel_seq])

    comb_seq = Dense(20, activation='relu')(merge_seq)
    comb_seq = Dropout(.25)(comb_seq)

    #comb_seq = concatenate([comb_seq, hl_channel_seq])

    #comb_seq = Dense(100, activation='relu')(comb_seq)
    comb_seq = Dense(100, activation='relu')(hl_channel_seq)
    comb_seq = Dropout(.25)(comb_seq)

    comb_seq = Dense(1, activation='sigmoid')(comb_seq)

    #combined_rnn = Model(inputs = [jet_channel_input, lep_channel_input], outputs = [comb_seq])
    combined_rnn = Model(inputs = [jet_channel_input, lep_channel_input, hl_channel_input], outputs = [comb_seq])



    # jet_channel.add(Masking(mask_value=-999, input_shape=JET_SHAPE, name='jet_masking'))
    # jet_channel.add(GRU(25, name='jet_gru'))
    # jet_channel.add(Dropout(0.3, name='jet_dropout'))

    # lep_channel.add(Masking(mask_value=-999, input_shape=LEP_SHAPE, name='lep_masking'))
    # lep_channel.add(GRU(10, name='lep_gru'))
    # lep_channel.add(Dropout(0.3, name='lep_dropout'))

    # merged_seq = Concatenate()([jet_channel.output,lep_channel.output])

    # #merged_seq = Flatten()(merged_seq)
    # merged_seq = Dense(150, activation='relu')(merged_seq)
    # merged_seq = Dropout(.25)(merged_seq)
    # merged_seq = Dense(100, activation='relu')(merged_seq)
    # merged_seq = Dropout(.25)(merged_seq)

    # merged_seq = Dense(1, activation='sigmoid')(merged_seq)

    # combined_rnn = Model([jet_channel.input,lep_channel.input], merged_seq)

    combined_rnn.summary()

    #opt = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.001, nesterov=False)
    opt = optimizers.SGD(lr=0.1, momentum=0.1, decay=0.001, nesterov=False)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = optimizers.Adamax(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)

    #loss = 'categorical_crossentropy'
    loss = 'binary_crossentropy'
    #loss = 'mean_squared_error'

    #combined_rnn.compile('adam', )
    #combined_rnn.compile(opt, loss, metrics=['binary_accuracy'])
    combined_rnn.compile(opt, loss, metrics=['accuracy'])

    #print(X_j_train, X_l_test, y_train)
    #print(X_hl_train, X_hl_test, y_train)

    print('Training...')

    combined_rnn.fit([X_j_train, X_l_train, X_hl_train], y_train, batch_size=arg['batchsize'],
                     callbacks = [
                         EarlyStopping(verbose=True, patience=25, monitor='val_loss'),
                         #ModelCheckpoint(opath + '/model/weights_e-{epoch:03d}_l-{val_loss:.3f}_a-{val_acc:.3f}.h5', monitor='val_loss', verbose=True, save_best_only=True)
                         ModelCheckpoint(opath + '/model/weights_e-{epoch:03d}.h5', monitor='val_loss', verbose=True, save_best_only=True)
                     ],
                     epochs=arg['epochs'],
                     validation_split = 0.2)

    if do_plot_model:
        plot_model(combined_rnn,
                   to_file=opath + 'model.png',
                   show_shapes=True,
                   show_layer_names=True)



    yhat_rnn = combined_rnn.predict([X_j_test, X_l_test, X_hl_test], verbose = True, batch_size = 512)


    #plot.training_history(training.history, opath)



    print('\nEvaluating training sample...')
    eval_train = combined_rnn.predict([X_j_train, X_l_train, X_hl_train], verbose = True, batch_size = 512)
    #print(eval_train, y_train)
    print()
    print('\nEvaluating test sample...')
    eval_test = combined_rnn.predict([X_j_test, X_l_test, X_hl_test], verbose = True, batch_size = 512)
    #print(eval_test, y_test)

    print('\n')

    roc(y_train, eval_train, opath, 'test')
    roc(y_test, eval_test, opath, 'train')
    plot.overtrain(eval_train, y_train, eval_test, y_test, opath)
    rootIO.save_training_reults(y_train, eval_train, y_test, eval_test, opath)


    # save the model and weights to files
    print('--- Saving model')
    combined_rnn.save(opath + '/model/complete.h5')
    combined_rnn.save_weights(opath + '/model/weights.h5')
    model_json = combined_rnn.to_json()
    with open(opath + '/model/model.json', 'w') as json_file:
        json_file.write(model_json)
    model_yaml = combined_rnn.to_yaml()
    with open(opath + '/model/model.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)



    print('\n\n')
    print('--- Finished training!')




def roc(labels, values, opath, label):

    # ROC curve from sklearn
    fpr, tpr, thresholds = roc_curve(labels,
                                     values,
                                     pos_label=None,
                                     sample_weight=None,
                                     drop_intermediate=True)

    #print(fpr)
    #print(tpr)
    #print(thresholds)

    print('ROC ' + label + ':')
    auroc = roc_auc_score(labels, values)
    print(auroc)

    plot.roc(fpr, tpr, auroc, opath, label)

'''
