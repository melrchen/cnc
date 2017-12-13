from __future__ import print_function
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json
import matplotlib.pylab as plt
import numpy as np
import classif_preprocessing as pp 

# batch_size = 100
# num_classes = 5
# epochs = 20

# # input image dimensions
# img_x, img_y = 40, 40

# # load the grayscale data set, which already splits into train and test sets for us
# (x_train, y_train), (x_test, y_test) = pp.load_data()

# print('Loaded data set.')

# # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
# x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
# x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
# input_shape = (img_x, img_y, 1)

# print('Reshaped into a 4D tensor.')


# # convert the data to the right type
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# print('Normalized and converted data to the right type.')

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])


# # Neat thing that lets you view your accuracy over time
# class AccuracyHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.acc = []
#         self.val_acc = []

#     def on_epoch_end(self, batch, logs={}):
#         self.acc.append(logs.get('acc'))
#         self.val_acc.append(logs.get('val_acc'))

# history = AccuracyHistory()

# print('======TRAINING======')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[history])

# print("=======VALIDATING=======")
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Validation loss:', score[0])
# print('Validation accuracy:', score[1])

# # Plot the accuracy vs. epochs
# plt.plot(range(1, 21), history.val_acc)
# plt.xlabel('Epochs')
# plt.ylabel('Validation Accuracy')
# plt.show()

# plt.plot(range(1, 21), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Training Accuracy')
# plt.show()

# # print("======SAVING======")
# # model_json = model.to_json()

# # # Serialize model to JSON
# # with open('model.json', 'w') as json_file:
# #     json_file.write(model_json)

# # # Serialize weights to HDF5
# # model.save_weights("model.h5")





if __name__ == '__main__':
    # Plot the accuracy vs. epochs
  val = [0.32651227537323446, 0.31070735349374656, 0.2963351572261137, 0.28496674141463113, 0.27576662249424877, 0.2679966539144516, 0.26003319288001342, 0.25296965679701638, 0.24639656394720078, 0.24001785323900335, 0.23334262125632343, 0.22677944556755178, 0.21984287176062078, 0.2130672011305304, 0.20624293518417022, 0.19898123776211457, 0.19161274503259099, 0.18419186507954316, 0.17639734157744577, 0.16869022083633087, 0.16035977926324396, 0.15243225895306645, 0.14392574438277414, 0.13637136185870452, 0.12889761806410902, 0.1215074941077653, 0.11479578408248284, 0.10857112837188385, 0.1030221745810088, 0.097603746854207099, 0.093176985926487862, 0.08920021574286853, 0.086071534849265038, 0.081912573426961899, 0.078783289912868953, 0.076301336507586873, 0.073849680450032734, 0.071961329811636138, 0.069927046404165374, 0.068383411449544573, 0.067069531692301523, 0.065447589820798707, 0.064209983300636791, 0.063230535234598553, 0.062545706901480175, 0.061248095267835784, 0.060498210656292295, 0.059775340973454362, 0.059139096452032816, 0.058305053807356781, 0.059033305658137095, 0.057420442866928434, 0.056616948479238677, 0.05598665160291335, 0.055529270540265477, 0.055625509908970666, 0.05481919909224791, 0.054976049481945878, 0.054574648356612993, 0.053645342040587872, 0.053479111808187818, 0.053102336276103466, 0.052822742063333004, 0.052756557898486364, 0.052674480111283416, 0.052731234580278397, 0.051891264560468053, 0.051989173845333213, 0.051687213427880234, 0.051254644551697898, 0.051410498097538948, 0.050861527924151981, 0.050817974678733772, 0.050528776996275955, 0.050418609753251076, 0.050501265836989176, 0.051010982407366526, 0.050012867569046861, 0.04972048399641233, 0.049989466829334986, 0.049563127295935858, 0.04973886983797831, 0.049313246546422732, 0.049252382096122295, 0.049095871146110928, 0.049012469237341601, 0.049136228530722505, 0.048835916663793957, 0.04871882322956534, 0.048728628522332978, 0.04884639207054587, 0.048439491759328282, 0.048356343060731888, 0.048226204207714868, 0.048227075268240535, 0.048234252916539419, 0.04852937983677668, 0.047982623471933251, 0.047926729325862491, 0.047803523018956184, 0.048119906664771193, 0.047693675712627524, 0.047952927868155873, 0.047685574740171432, 0.047921831116956824, 0.048281544600339496, 0.047505120911142402, 0.047403103507616944, 0.047617452876532781, 0.048384503516204214, 0.047393811318804234, 0.047164322698817533, 0.047118319746326, 0.047793031505802101, 0.048764299382181728, 0.047314641120679235, 0.04765768360127421, 0.048287180416724258, 0.047039623317473078, 0.048942251350073254, 0.047466423472060877, 0.048306782017735875, 0.046595018904875309, 0.046605879988740474, 0.046704439555897433, 0.046487766263239527, 0.046511966077720415, 0.046382772791035032, 0.046555462128975815, 0.046426516464527917, 0.046358567279051334, 0.046317326014532763, 0.046274145517279118, 0.047635919688379061, 0.046566263717763567, 0.046393744537935537, 0.047034860862528574, 0.046124480226460624, 0.046171607997487575, 0.046116110594833601, 0.046583285116974044, 0.046376815065741539, 0.046185176819562912, 0.046432874329826408, 0.045901535407585257, 0.046148592934888953, 0.045830283971393812, 0.046541512341183776, 0.046060621081029665, 0.046354383778046161, 0.047858214224962628, 0.047086963649181759, 0.04587702255915193, 0.04633880910627982, 0.045731820878298843, 0.047132968573885804, 0.045692745486603063, 0.046004477559643632, 0.045718675972345996, 0.046143808676039469, 0.045604257406119036, 0.045623460565419757, 0.047370330814053029, 0.046022033209309858, 0.045740848461932992, 0.045453830019516105, 0.045492057962452662, 0.047542046207715481, 0.046823180225842143, 0.045334251761874729, 0.04535146419178037, 0.045704575803350002, 0.045945386237957901, 0.045304548050112581, 0.045348551124334335, 0.045726770042058301, 0.046480430290102959, 0.045445679193910432, 0.045450534123708218, 0.045501249404076266, 0.04525220470831675, 0.04512004225569613, 0.045276707805254883, 0.045232670052963143, 0.047605322893051541, 0.045381267171572238, 0.04528783426127013, 0.045063174932318574, 0.04615487048731131, 0.04583964957033887, 0.045104812721119207, 0.045063841485363597, 0.045682005748590999, 0.04561838091296308, 0.046116277027656048, 0.04501210218843292, 0.045129104681751304, 0.044980588807340932, 0.044931886279407668, 0.044918783118619639, 0.045544182388659787, 0.045742682783919221, 0.045082672608687598, 0.045075102654450083, 0.044869625447865796, 0.045044753152657956, 0.045082820579409599, 0.044938131016405189, 0.044787701000185573, 0.045007156975129074, 0.045547986786593407, 0.045387789716615397, 0.045203632127274483, 0.044729093473185509, 0.044960636262069732, 0.045017864883822555, 0.045501038870390725, 0.044698167482719701, 0.044876792632481628, 0.04482267906560617, 0.044918112018529105, 0.044791956377380038, 0.045141077874337923, 0.045696465627235526, 0.044747857541284138, 0.044869081500698536, 0.044908067649778199, 0.044968478929470569, 0.045431751210023373, 0.04459525507820003, 0.044997733305482304, 0.044567906571661722, 0.045647586750633573, 0.045108925770310795, 0.045093397609889507, 0.044626791797140068, 0.045146319357787859, 0.04479347388533985, 0.044807009453720903, 0.045539419714580566, 0.04454214854494614, 0.044621512512950337, 0.044913559165947581, 0.045641247730921299, 0.044411390198065955, 0.044794033884125596, 0.045112842529573864, 0.044727557941394692, 0.045688847408575171, 0.04471103901810506, 0.044859789092751109, 0.045544560779543483, 0.045495075447594416, 0.04444341383436147, 0.045230223742478037, 0.047471298562253225, 0.045121990253820139, 0.044338764184538054, 0.044435700949500591, 0.044402874239227348, 0.044379410826984572, 0.045675583183765411, 0.045164912774720616, 0.044841104022720284, 0.044285439393099618, 0.044346850639318716, 0.044270066106144118, 0.044791858205023932, 0.045127273646785933, 0.045410203911802345, 0.044496261679074341, 0.04455916890326668, 0.045113895910189432, 0.044430779950583685, 0.044883606736274326, 0.045133937369374669, 0.044685445068513643, 0.044234610491377467, 0.044304339489077818, 0.04424980458091287, 0.045190862434751844, 0.044179495369248534, 0.044506145367289293, 0.044219299493467104, 0.044238554587697279, 0.049473204595201159, 0.044666087200098181, 0.044159027145189399, 0.044330307227723742, 0.045460142030873722, 0.044140266911948431, 0.044170162311809903, 0.044159775161567855, 0.044673669151961803, 0.044219685444498766, 0.044262788203709269, 0.044834989999585295, 0.04606878067202428, 0.044304285527152175, 0.045268188066342301, 0.045124068965806681, 0.044124463682665545, 0.044114116798428929, 0.044815890545792436, 0.044382329153663969, 0.044071509831530208, 0.044340953230857849, 0.044064733144991541, 0.044382330468472314, 0.043972511940142685, 0.04452871980474276, 0.044551959535216581, 0.044174495974884313, 0.044063531519735563, 0.044934549149783218, 0.044279974885284901, 0.044787478030604476, 0.044086532667279243, 0.044481643968645262, 0.044043845880557507, 0.044611666954177266, 0.044455277076100599, 0.0459878109395504, 0.045289522256044781, 0.044111734803985149, 0.044477185727480578, 0.044285911902347035, 0.044617385986973256, 0.04399284994339242, 0.04397609386154834, 0.044416650417534745, 0.043961588786367106, 0.044338323285474497, 0.04419384420137195, 0.045214594889651329, 0.044198570279952359, 0.045287269386736786, 0.044465948005809504, 0.043844493927762788, 0.043835978188058906, 0.043922875953071261, 0.043979896068134731, 0.044044912573607528, 0.043898178209715033, 0.04391560298116768, 0.044048351126120371, 0.04422104271019206, 0.044265316361013579, 0.044761036687037521, 0.044730125214247143, 0.045622353442013264, 0.044144928455352783, 0.044074382906889212, 0.043950555791311408, 0.044353745658608043, 0.044580247965367401, 0.044911564875613243, 0.044139771886608177, 0.044220580335925609, 0.044128625872818864, 0.046559962937060523, 0.045061508029261059, 0.044712543268414104, 0.044027713784838426, 0.044102291193078548, 0.04386071444434278, 0.044451696798205376, 0.044962596312603527, 0.047015116604812002, 0.044167120886199618, 0.043820042272700983, 0.043955097248887312, 0.044638047816560548, 0.045183405718382669, 0.044359721681650949, 0.044597477542565149, 0.043808781432316583, 0.044951363850165814, 0.045270243823966556, 0.045213527484413457, 0.043830546714803749, 0.043811713947969323, 0.044226895086467266, 0.043694240643697625, 0.04382374756695593, 0.044222294079030264, 0.04538087360560894, 0.044408316176165551, 0.043822868014959729, 0.044180912020451879, 0.043689617503653556, 0.043929055990541685, 0.043721824007875776, 0.04482063519604066, 0.043916391920955741, 0.044148678014821863, 0.043746313189758974, 0.043867454973652083, 0.044286857359111309, 0.043642670628340804]
  tr = [0.35882512614130974, 0.31562487065792083, 0.29772167131304739, 0.28594625353813169, 0.27677816405892375, 0.26862082690000533, 0.26073547117412088, 0.25367767870426178, 0.24696511521935463, 0.24037200666964054, 0.23377159945666789, 0.22710708916187286, 0.22034058295190334, 0.21347333818674089, 0.2064690662920475, 0.19931743122637272, 0.19197999782860278, 0.1845390125364065, 0.17688873574137687, 0.16905582047998904, 0.16108583375811578, 0.15307737909257413, 0.14519473299384117, 0.13743022993206977, 0.12997239336371422, 0.1228653022274375, 0.11621271934360265, 0.11003493268042802, 0.10437036126852035, 0.099268736876547331, 0.094556005597114565, 0.090382095538079743, 0.086664060875773427, 0.08338451825082302, 0.080431789234280585, 0.077819034159183509, 0.075497519802302127, 0.073415649607777592, 0.071565687004476783, 0.069916953612118957, 0.068393366318196064, 0.067066685520112509, 0.065833782330155369, 0.064784451834857468, 0.063726908881217237, 0.062830735556781286, 0.062043052967637777, 0.061264028083533047, 0.060555700305849315, 0.059956068620085713, 0.059326115492731334, 0.058811871539801358, 0.058266994543373585, 0.057874069996178147, 0.057396721858531237, 0.056984098479151729, 0.05660519389435649, 0.056196021400392059, 0.055887236353009941, 0.055581484846770765, 0.055275728534907101, 0.055057309642434119, 0.054776767622679474, 0.054482809677720068, 0.054224352445453407, 0.054010689109563831, 0.053822478018701075, 0.053571597654372451, 0.053371721021831034, 0.053191800713539121, 0.053055276721715929, 0.052869499307125804, 0.052711281944066289, 0.052592798620462421, 0.052382589345797897, 0.052263037916272875, 0.05218742448836565, 0.051985105276107786, 0.051935446914285421, 0.051785369757562878, 0.051676894500851632, 0.051568158660084011, 0.051440312024205924, 0.051320965047925708, 0.051258910708129407, 0.05117970984429121, 0.051067099794745448, 0.050985984327271583, 0.050856735818088056, 0.050809820955619214, 0.050685839150100949, 0.050651401458308098, 0.050550646688789128, 0.050570712238550183, 0.050419321581721306, 0.050340839177370068, 0.050312919449061154, 0.050207686312496662, 0.050118799358606338, 0.050082667488604786, 0.050055949939414861, 0.049950962411239745, 0.049847232997417452, 0.049872130025178193, 0.049816351477056743, 0.049745747130364178, 0.049686677502468225, 0.049680207464843987, 0.049566218536347149, 0.049484990648925302, 0.04952442856505513, 0.049483545646071436, 0.049437981052324179, 0.049362671850249173, 0.049284151131287215, 0.049244441520422698, 0.049257043078541757, 0.049248415995389226, 0.04912041300907731, 0.049150166660547258, 0.049127675965428351, 0.049056857638061044, 0.049066458418965342, 0.048965533412992951, 0.048924300558865069, 0.048896994888782498, 0.048898871308192614, 0.048838633969426157, 0.048840286610648036, 0.048831724403426049, 0.04878633189946413, 0.048769245967268941, 0.048766524540260432, 0.048689259700477126, 0.048634784929454326, 0.048721594456583264, 0.048619764493778347, 0.048669984694570304, 0.048507669651880858, 0.048516899067908528, 0.048454658715054395, 0.048404079759493471, 0.048508788850158455, 0.048457147292792796, 0.048394001154229045, 0.048441299721598628, 0.048309767190366983, 0.048321116222068666, 0.048237881623208523, 0.048214519908651707, 0.048289076685905458, 0.048304912429302932, 0.048264490282163022, 0.04823237047530711, 0.048158692745491866, 0.048162398599088191, 0.048099331557750702, 0.048141522938385604, 0.048126700939610599, 0.048059108862653376, 0.048142216084524986, 0.048089440027251837, 0.047986633786931632, 0.048024004455655814, 0.048057487625628711, 0.048094705352559684, 0.047907374631613493, 0.047925514495000245, 0.048021323522552846, 0.047895140144973994, 0.047942491080611944, 0.047854846082627772, 0.047879869034513828, 0.047812720434740184, 0.047911656973883512, 0.047871571648865935, 0.047741777868941425, 0.047747370693832634, 0.047819013623520729, 0.047764552654698492, 0.047801925381645557, 0.047711771624162791, 0.047721936898306012, 0.047700752113014458, 0.047708543604239824, 0.047718244744464754, 0.047740742210298774, 0.047632507551461457, 0.04762653830461204, 0.047674220576882365, 0.047629541931673881, 0.047550564510747788, 0.047588751344010236, 0.047603359213098884, 0.047608517166227105, 0.047541618179529906, 0.0475914303958416, 0.047538868347182871, 0.0475256138201803, 0.047586620301008221, 0.047602872848510745, 0.047520812153816226, 0.047583763757720589, 0.047497682841494679, 0.047467616721987724, 0.04745290166698396, 0.047437221538275481, 0.047500328347086909, 0.047422499880194667, 0.047386608319357038, 0.047423516744747755, 0.04741366192698479, 0.047354914778843522, 0.047462881626561287, 0.04738111551851034, 0.047406652895733717, 0.047396648321300743, 0.047352918637916443, 0.047366249179467558, 0.04735203971154988, 0.047409442961215971, 0.0473903832398355, 0.047420088406652211, 0.047248315745964643, 0.047317767702043059, 0.047298859180882571, 0.047235599150881172, 0.047301888614892956, 0.047226790403947234, 0.047302857693284753, 0.04713998980820179, 0.047184869945049286, 0.047219879375770685, 0.047171234898269174, 0.047235427340492603, 0.047242924338206649, 0.047243176111951474, 0.047229295410215852, 0.047167389811947943, 0.047127404063940045, 0.047163019673898819, 0.047172324834391474, 0.047178735174238678, 0.047094274368137123, 0.047178213633596894, 0.047184956930577754, 0.047151529137045144, 0.047057239143177866, 0.047144340211525561, 0.047032620245590809, 0.047067131567746404, 0.047148886620998386, 0.047103462396189573, 0.047129356618970636, 0.047046223366633055, 0.04708986525423825, 0.047060173433274032, 0.047139510130509737, 0.047102888328954574, 0.04700147391296923, 0.047138463398441674, 0.047102593667805198, 0.04702820785343647, 0.04707241642288864, 0.04698014544323087, 0.047068848107010126, 0.046950517967343328, 0.047118524042889479, 0.046923027122393247, 0.046940793730318543, 0.046972495838999748, 0.046981766959652303, 0.046987724592909216, 0.046945785945281385, 0.047026208406314253, 0.047053376939147712, 0.046961647477000955, 0.046931289806962012, 0.046908059045672419, 0.046977246664464477, 0.046957476092502476, 0.046951907509937885, 0.046834606835618617, 0.047055975347757337, 0.047027310328558086, 0.046969745075330138, 0.046834239466115833, 0.046855896413326263, 0.046960678128525613, 0.046933919917792083, 0.046937438426539302, 0.046841835761442782, 0.046940363524481651, 0.046833137534558776, 0.046910756630823017, 0.046842830860987304, 0.046849015597254039, 0.04679619338363409, 0.046885415883734821, 0.046843720497563482, 0.046851163320243361, 0.046814871849492193, 0.046869790982455015, 0.046752466177567842, 0.046900922143831847, 0.046789353461936113, 0.046782585019245745, 0.046705100191757082, 0.046782352821901442, 0.046813960168510672, 0.046729815052822234, 0.046771897412836549, 0.046815781407058242, 0.046813041605055332, 0.046779943052679303, 0.046743412753567103, 0.046802103761583565, 0.046803019577637314, 0.046764691118150951, 0.046741315685212614, 0.04668093259446323, 0.046738896323367952, 0.046778271514922379, 0.046733949109911917, 0.046699885772541166, 0.046760524362325667, 0.04675059612840414, 0.046691149547696112, 0.046693468960002066, 0.046710498696193099, 0.046757359215989711, 0.046812637401744725, 0.046639470867812637, 0.046694522751495243, 0.046696389392018318, 0.046611064318567513, 0.046743468130007383, 0.046668067434802653, 0.046651075035333632, 0.046591591099277137, 0.046790935937315224, 0.046654042443260553, 0.046660755965858697, 0.046631870111450556, 0.046654235916212199, 0.046693767076358202, 0.046661391127854585, 0.046655910611152651, 0.046672447230666873, 0.046604845337569714, 0.046604664232581854, 0.046677264561876652, 0.046621334133669735, 0.046598031362518666, 0.046627026507630946, 0.046676972005516293, 0.046819293601438404, 0.046581319449469444, 0.046586080444976689, 0.04663800840266049, 0.046589885102584959, 0.046598241860046984, 0.046701179286465049, 0.046554138632491228, 0.046514147603884337, 0.046564752785488966, 0.046661897897720336, 0.046667625727131963, 0.046565095158293844, 0.046584957689046858, 0.046662871669977901, 0.046580201368778947, 0.046547829024493698, 0.046595602529123425, 0.046587725747376678, 0.04653128804638982, 0.046603066576644779, 0.046592202745378015, 0.046634396864101292, 0.046475104596465824, 0.046544580487534404, 0.046635063057765366, 0.046629955489188434, 0.046547596259042617, 0.046531936172395948, 0.046512744780629875, 0.04656281421892345, 0.046567532205954193, 0.046536903884261849, 0.046491462578997014, 0.046453920695930719, 0.046590664619579912, 0.046428448054939508, 0.046452366495504972, 0.046521512232720851, 0.046635828940197827, 0.046437910282984378, 0.046626414693892004, 0.046471915598958732, 0.046585220880806445]

  plt.plot(range(1, 401), val)
  plt.xlabel('Epochs')
  plt.ylabel('Validation Loss')
  plt.show()

  plt.plot(range(1, 401), tr)
  plt.xlabel('Epochs')
  plt.ylabel('Training Loss')
  plt.show()
    # score = loaded_model.evaluate(X, Y, verbose=0)