¯
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18÷ð

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:
*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:
*
dtype0

dense_9716/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 ±	
*"
shared_namedense_9716/kernel
y
%dense_9716/kernel/Read/ReadVariableOpReadVariableOpdense_9716/kernel* 
_output_shapes
:
 ±	
*
dtype0
v
dense_9716/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_9716/bias
o
#dense_9716/bias/Read/ReadVariableOpReadVariableOpdense_9716/bias*
_output_shapes
:
*
dtype0
~
dense_9717/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*"
shared_namedense_9717/kernel
w
%dense_9717/kernel/Read/ReadVariableOpReadVariableOpdense_9717/kernel*
_output_shapes

:
*
dtype0
v
dense_9717/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_9717/bias
o
#dense_9717/bias/Read/ReadVariableOpReadVariableOpdense_9717/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:
*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_9716/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 ±	
*)
shared_nameAdam/dense_9716/kernel/m

,Adam/dense_9716/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9716/kernel/m* 
_output_shapes
:
 ±	
*
dtype0

Adam/dense_9716/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_9716/bias/m
}
*Adam/dense_9716/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9716/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_9717/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*)
shared_nameAdam/dense_9717/kernel/m

,Adam/dense_9717/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9717/kernel/m*
_output_shapes

:
*
dtype0

Adam/dense_9717/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_9717/bias/m
}
*Adam/dense_9717/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9717/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:
*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_9716/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 ±	
*)
shared_nameAdam/dense_9716/kernel/v

,Adam/dense_9716/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9716/kernel/v* 
_output_shapes
:
 ±	
*
dtype0

Adam/dense_9716/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_9716/bias/v
}
*Adam/dense_9716/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9716/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_9717/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*)
shared_nameAdam/dense_9717/kernel/v

,Adam/dense_9717/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9717/kernel/v*
_output_shapes

:
*
dtype0

Adam/dense_9717/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_9717/bias/v
}
*Adam/dense_9717/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9717/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ö*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±*
value§*B¤* B*

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
|
_inbound_nodes

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
f
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
f
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
|
_inbound_nodes

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
|
$_inbound_nodes

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
¬
+iter

,beta_1

-beta_2
	.decay
/learning_ratemYmZm[m\%m]&m^v_v`vavb%vc&vd
 
*
0
1
2
3
%4
&5
*
0
1
2
3
%4
&5
­
0metrics

1layers
2non_trainable_variables
3layer_regularization_losses
regularization_losses
trainable_variables
		variables
4layer_metrics
 
 
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
5metrics

6layers
7non_trainable_variables
8layer_regularization_losses
regularization_losses
trainable_variables
	variables
9layer_metrics
 
 
 
 
­
:metrics

;layers
<non_trainable_variables
=layer_regularization_losses
regularization_losses
trainable_variables
	variables
>layer_metrics
 
 
 
 
­
?metrics

@layers
Anon_trainable_variables
Blayer_regularization_losses
regularization_losses
trainable_variables
	variables
Clayer_metrics
 
][
VARIABLE_VALUEdense_9716/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9716/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Dmetrics

Elayers
Fnon_trainable_variables
Glayer_regularization_losses
 regularization_losses
!trainable_variables
"	variables
Hlayer_metrics
 
][
VARIABLE_VALUEdense_9717/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9717/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
­
Imetrics

Jlayers
Knon_trainable_variables
Llayer_regularization_losses
'regularization_losses
(trainable_variables
)	variables
Mlayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ptotal
	Qcount
R	variables
S	keras_api
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_9716/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_9716/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_9717/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_9717/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_9716/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_9716/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_9717/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_9717/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_5_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿúú
­
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_5_inputconv2d_5/kernelconv2d_5/biasdense_9716/kerneldense_9716/biasdense_9717/kerneldense_9717/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_833268
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ã

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp%dense_9716/kernel/Read/ReadVariableOp#dense_9716/bias/Read/ReadVariableOp%dense_9717/kernel/Read/ReadVariableOp#dense_9717/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp,Adam/dense_9716/kernel/m/Read/ReadVariableOp*Adam/dense_9716/bias/m/Read/ReadVariableOp,Adam/dense_9717/kernel/m/Read/ReadVariableOp*Adam/dense_9717/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp,Adam/dense_9716/kernel/v/Read/ReadVariableOp*Adam/dense_9716/bias/v/Read/ReadVariableOp,Adam/dense_9717/kernel/v/Read/ReadVariableOp*Adam/dense_9717/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_833533
¢
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasdense_9716/kerneldense_9716/biasdense_9717/kerneldense_9717/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/dense_9716/kernel/mAdam/dense_9716/bias/mAdam/dense_9717/kernel/mAdam/dense_9717/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/dense_9716/kernel/vAdam/dense_9716/bias/vAdam/dense_9717/kernel/vAdam/dense_9717/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_833624æü
¦
Ý
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833188

inputs
conv2d_5_833170
conv2d_5_833172
dense_9716_833177
dense_9716_833179
dense_9717_833182
dense_9717_833184
identity¢ conv2d_5/StatefulPartitionedCall¢"dense_9716/StatefulPartitionedCall¢"dense_9717/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_833170conv2d_5_833172*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_8330572"
 conv2d_5/StatefulPartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_8330362!
max_pooling2d_4/PartitionedCallû
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8330802
flatten_3/PartitionedCallº
"dense_9716/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9716_833177dense_9716_833179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9716_layer_call_and_return_conditional_losses_8330992$
"dense_9716/StatefulPartitionedCallÃ
"dense_9717/StatefulPartitionedCallStatefulPartitionedCall+dense_9716/StatefulPartitionedCall:output:0dense_9717_833182dense_9717_833184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9717_layer_call_and_return_conditional_losses_8331262$
"dense_9717/StatefulPartitionedCallì
IdentityIdentity+dense_9717/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall#^dense_9716/StatefulPartitionedCall#^dense_9717/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"dense_9716/StatefulPartitionedCall"dense_9716/StatefulPartitionedCall2H
"dense_9717/StatefulPartitionedCall"dense_9717/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
¿
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_833080

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ X 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ||
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||

 
_user_specified_nameinputs
¾
å
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833164
conv2d_5_input
conv2d_5_833146
conv2d_5_833148
dense_9716_833153
dense_9716_833155
dense_9717_833158
dense_9717_833160
identity¢ conv2d_5/StatefulPartitionedCall¢"dense_9716/StatefulPartitionedCall¢"dense_9717/StatefulPartitionedCall¦
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_833146conv2d_5_833148*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_8330572"
 conv2d_5/StatefulPartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_8330362!
max_pooling2d_4/PartitionedCallû
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8330802
flatten_3/PartitionedCallº
"dense_9716/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9716_833153dense_9716_833155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9716_layer_call_and_return_conditional_losses_8330992$
"dense_9716/StatefulPartitionedCallÃ
"dense_9717/StatefulPartitionedCallStatefulPartitionedCall+dense_9716/StatefulPartitionedCall:output:0dense_9717_833158dense_9717_833160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9717_layer_call_and_return_conditional_losses_8331262$
"dense_9717/StatefulPartitionedCallì
IdentityIdentity+dense_9717/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall#^dense_9716/StatefulPartitionedCall#^dense_9717/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"dense_9716/StatefulPartitionedCall"dense_9716/StatefulPartitionedCall2H
"dense_9717/StatefulPartitionedCall"dense_9717/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
(
_user_specified_nameconv2d_5_input
¤s
ü
"__inference__traced_restore_833624
file_prefix$
 assignvariableop_conv2d_5_kernel$
 assignvariableop_1_conv2d_5_bias(
$assignvariableop_2_dense_9716_kernel&
"assignvariableop_3_dense_9716_bias(
$assignvariableop_4_dense_9717_kernel&
"assignvariableop_5_dense_9717_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1.
*assignvariableop_15_adam_conv2d_5_kernel_m,
(assignvariableop_16_adam_conv2d_5_bias_m0
,assignvariableop_17_adam_dense_9716_kernel_m.
*assignvariableop_18_adam_dense_9716_bias_m0
,assignvariableop_19_adam_dense_9717_kernel_m.
*assignvariableop_20_adam_dense_9717_bias_m.
*assignvariableop_21_adam_conv2d_5_kernel_v,
(assignvariableop_22_adam_conv2d_5_bias_v0
,assignvariableop_23_adam_dense_9716_kernel_v.
*assignvariableop_24_adam_dense_9716_bias_v0
,assignvariableop_25_adam_dense_9717_kernel_v.
*assignvariableop_26_adam_dense_9717_bias_v
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¸
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_9716_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_9716_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_9717_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_9717_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¡
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15²
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_5_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_conv2d_5_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17´
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_9716_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18²
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_9716_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19´
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_9717_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20²
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_9717_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21²
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_5_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_5_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23´
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_dense_9716_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_9716_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25´
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_dense_9717_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26²
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_9717_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp°
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27£
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¾
å
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833143
conv2d_5_input
conv2d_5_833068
conv2d_5_833070
dense_9716_833110
dense_9716_833112
dense_9717_833137
dense_9717_833139
identity¢ conv2d_5/StatefulPartitionedCall¢"dense_9716/StatefulPartitionedCall¢"dense_9717/StatefulPartitionedCall¦
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_833068conv2d_5_833070*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_8330572"
 conv2d_5/StatefulPartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_8330362!
max_pooling2d_4/PartitionedCallû
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8330802
flatten_3/PartitionedCallº
"dense_9716/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9716_833110dense_9716_833112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9716_layer_call_and_return_conditional_losses_8330992$
"dense_9716/StatefulPartitionedCallÃ
"dense_9717/StatefulPartitionedCallStatefulPartitionedCall+dense_9716/StatefulPartitionedCall:output:0dense_9717_833137dense_9717_833139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9717_layer_call_and_return_conditional_losses_8331262$
"dense_9717/StatefulPartitionedCallì
IdentityIdentity+dense_9717/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall#^dense_9716/StatefulPartitionedCall#^dense_9717/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"dense_9716/StatefulPartitionedCall"dense_9716/StatefulPartitionedCall2H
"dense_9717/StatefulPartitionedCall"dense_9717/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
(
_user_specified_nameconv2d_5_input

g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_833036

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

É
0__inference_sequential_4858_layer_call_fn_833203
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_4858_layer_call_and_return_conditional_losses_8331882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
(
_user_specified_nameconv2d_5_input
	
¬
D__inference_conv2d_5_layer_call_and_return_conditional_losses_833057

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿúú:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
¦
Ý
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833226

inputs
conv2d_5_833208
conv2d_5_833210
dense_9716_833215
dense_9716_833217
dense_9717_833220
dense_9717_833222
identity¢ conv2d_5/StatefulPartitionedCall¢"dense_9716/StatefulPartitionedCall¢"dense_9717/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_833208conv2d_5_833210*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_8330572"
 conv2d_5/StatefulPartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_8330362!
max_pooling2d_4/PartitionedCallû
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8330802
flatten_3/PartitionedCallº
"dense_9716/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9716_833215dense_9716_833217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9716_layer_call_and_return_conditional_losses_8330992$
"dense_9716/StatefulPartitionedCallÃ
"dense_9717/StatefulPartitionedCallStatefulPartitionedCall+dense_9716/StatefulPartitionedCall:output:0dense_9717_833220dense_9717_833222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9717_layer_call_and_return_conditional_losses_8331262$
"dense_9717/StatefulPartitionedCallì
IdentityIdentity+dense_9717/StatefulPartitionedCall:output:0!^conv2d_5/StatefulPartitionedCall#^dense_9716/StatefulPartitionedCall#^dense_9717/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2H
"dense_9716/StatefulPartitionedCall"dense_9716/StatefulPartitionedCall2H
"dense_9717/StatefulPartitionedCall"dense_9717/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
ÿ
Á
0__inference_sequential_4858_layer_call_fn_833358

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_4858_layer_call_and_return_conditional_losses_8332262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
ÿ$
Á
!__inference__wrapped_model_833030
conv2d_5_input;
7sequential_4858_conv2d_5_conv2d_readvariableop_resource<
8sequential_4858_conv2d_5_biasadd_readvariableop_resource=
9sequential_4858_dense_9716_matmul_readvariableop_resource>
:sequential_4858_dense_9716_biasadd_readvariableop_resource=
9sequential_4858_dense_9717_matmul_readvariableop_resource>
:sequential_4858_dense_9717_biasadd_readvariableop_resource
identityà
.sequential_4858/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7sequential_4858_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype020
.sequential_4858/conv2d_5/Conv2D/ReadVariableOpù
sequential_4858/conv2d_5/Conv2DConv2Dconv2d_5_input6sequential_4858/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*
paddingVALID*
strides
2!
sequential_4858/conv2d_5/Conv2D×
/sequential_4858/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8sequential_4858_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype021
/sequential_4858/conv2d_5/BiasAdd/ReadVariableOpî
 sequential_4858/conv2d_5/BiasAddBiasAdd(sequential_4858/conv2d_5/Conv2D:output:07sequential_4858/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2"
 sequential_4858/conv2d_5/BiasAdd­
sequential_4858/conv2d_5/ReluRelu)sequential_4858/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
sequential_4858/conv2d_5/Relu÷
'sequential_4858/max_pooling2d_4/MaxPoolMaxPool+sequential_4858/conv2d_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
*
ksize
*
paddingVALID*
strides
2)
'sequential_4858/max_pooling2d_4/MaxPool
sequential_4858/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ X 2!
sequential_4858/flatten_3/Constá
!sequential_4858/flatten_3/ReshapeReshape0sequential_4858/max_pooling2d_4/MaxPool:output:0(sequential_4858/flatten_3/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2#
!sequential_4858/flatten_3/Reshapeà
0sequential_4858/dense_9716/MatMul/ReadVariableOpReadVariableOp9sequential_4858_dense_9716_matmul_readvariableop_resource* 
_output_shapes
:
 ±	
*
dtype022
0sequential_4858/dense_9716/MatMul/ReadVariableOpè
!sequential_4858/dense_9716/MatMulMatMul*sequential_4858/flatten_3/Reshape:output:08sequential_4858/dense_9716/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2#
!sequential_4858/dense_9716/MatMulÝ
1sequential_4858/dense_9716/BiasAdd/ReadVariableOpReadVariableOp:sequential_4858_dense_9716_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1sequential_4858/dense_9716/BiasAdd/ReadVariableOpí
"sequential_4858/dense_9716/BiasAddBiasAdd+sequential_4858/dense_9716/MatMul:product:09sequential_4858/dense_9716/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2$
"sequential_4858/dense_9716/BiasAdd©
sequential_4858/dense_9716/ReluRelu+sequential_4858/dense_9716/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2!
sequential_4858/dense_9716/ReluÞ
0sequential_4858/dense_9717/MatMul/ReadVariableOpReadVariableOp9sequential_4858_dense_9717_matmul_readvariableop_resource*
_output_shapes

:
*
dtype022
0sequential_4858/dense_9717/MatMul/ReadVariableOpë
!sequential_4858/dense_9717/MatMulMatMul-sequential_4858/dense_9716/Relu:activations:08sequential_4858/dense_9717/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_4858/dense_9717/MatMulÝ
1sequential_4858/dense_9717/BiasAdd/ReadVariableOpReadVariableOp:sequential_4858_dense_9717_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4858/dense_9717/BiasAdd/ReadVariableOpí
"sequential_4858/dense_9717/BiasAddBiasAdd+sequential_4858/dense_9717/MatMul:product:09sequential_4858/dense_9717/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_4858/dense_9717/BiasAdd²
"sequential_4858/dense_9717/SoftmaxSoftmax+sequential_4858/dense_9717/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_4858/dense_9717/Softmax
IdentityIdentity,sequential_4858/dense_9717/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú:::::::a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
(
_user_specified_nameconv2d_5_input
³
®
F__inference_dense_9717_layer_call_and_return_conditional_losses_833420

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¨
F
*__inference_flatten_3_layer_call_fn_833389

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_8330802
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ||
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||

 
_user_specified_nameinputs
®>
¢
__inference__traced_save_833533
file_prefix.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop0
,savev2_dense_9716_kernel_read_readvariableop.
*savev2_dense_9716_bias_read_readvariableop0
,savev2_dense_9717_kernel_read_readvariableop.
*savev2_dense_9717_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop7
3savev2_adam_dense_9716_kernel_m_read_readvariableop5
1savev2_adam_dense_9716_bias_m_read_readvariableop7
3savev2_adam_dense_9717_kernel_m_read_readvariableop5
1savev2_adam_dense_9717_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop7
3savev2_adam_dense_9716_kernel_v_read_readvariableop5
1savev2_adam_dense_9716_bias_v_read_readvariableop7
3savev2_adam_dense_9717_kernel_v_read_readvariableop5
1savev2_adam_dense_9717_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_02b29a1c4e8547c18aa89867cd45c548/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÀ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop,savev2_dense_9716_kernel_read_readvariableop*savev2_dense_9716_bias_read_readvariableop,savev2_dense_9717_kernel_read_readvariableop*savev2_dense_9717_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop3savev2_adam_dense_9716_kernel_m_read_readvariableop1savev2_adam_dense_9716_bias_m_read_readvariableop3savev2_adam_dense_9717_kernel_m_read_readvariableop1savev2_adam_dense_9717_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop3savev2_adam_dense_9716_kernel_v_read_readvariableop1savev2_adam_dense_9716_bias_v_read_readvariableop3savev2_adam_dense_9717_kernel_v_read_readvariableop1savev2_adam_dense_9717_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ù
_input_shapesÇ
Ä: :
:
:
 ±	
:
:
:: : : : : : : : : :
:
:
 ±	
:
:
::
:
:
 ±	
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
 ±	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
 ±	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:&"
 
_output_shapes
:
 ±	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: 
±
®
F__inference_dense_9716_layer_call_and_return_conditional_losses_833400

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 ±	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ±	:::Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	
 
_user_specified_nameinputs
í

K__inference_sequential_4858_layer_call_and_return_conditional_losses_833324

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource-
)dense_9716_matmul_readvariableop_resource.
*dense_9716_biasadd_readvariableop_resource-
)dense_9717_matmul_readvariableop_resource.
*dense_9717_biasadd_readvariableop_resource
identity°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02 
conv2d_5/Conv2D/ReadVariableOpÁ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*
paddingVALID*
strides
2
conv2d_5/Conv2D§
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp®
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
conv2d_5/BiasAdd}
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
conv2d_5/ReluÇ
max_pooling2d_4/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ X 2
flatten_3/Const¡
flatten_3/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2
flatten_3/Reshape°
 dense_9716/MatMul/ReadVariableOpReadVariableOp)dense_9716_matmul_readvariableop_resource* 
_output_shapes
:
 ±	
*
dtype02"
 dense_9716/MatMul/ReadVariableOp¨
dense_9716/MatMulMatMulflatten_3/Reshape:output:0(dense_9716/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9716/MatMul­
!dense_9716/BiasAdd/ReadVariableOpReadVariableOp*dense_9716_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!dense_9716/BiasAdd/ReadVariableOp­
dense_9716/BiasAddBiasAdddense_9716/MatMul:product:0)dense_9716/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9716/BiasAddy
dense_9716/ReluReludense_9716/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9716/Relu®
 dense_9717/MatMul/ReadVariableOpReadVariableOp)dense_9717_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02"
 dense_9717/MatMul/ReadVariableOp«
dense_9717/MatMulMatMuldense_9716/Relu:activations:0(dense_9717/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9717/MatMul­
!dense_9717/BiasAdd/ReadVariableOpReadVariableOp*dense_9717_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_9717/BiasAdd/ReadVariableOp­
dense_9717/BiasAddBiasAdddense_9717/MatMul:product:0)dense_9717/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9717/BiasAdd
dense_9717/SoftmaxSoftmaxdense_9717/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9717/Softmaxp
IdentityIdentitydense_9717/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú:::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
á

+__inference_dense_9717_layer_call_fn_833429

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9717_layer_call_and_return_conditional_losses_8331262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

É
0__inference_sequential_4858_layer_call_fn_833241
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_4858_layer_call_and_return_conditional_losses_8332262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
(
_user_specified_nameconv2d_5_input
³
®
F__inference_dense_9717_layer_call_and_return_conditional_losses_833126

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
¬
D__inference_conv2d_5_layer_call_and_return_conditional_losses_833369

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿúú:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs

~
)__inference_conv2d_5_layer_call_fn_833378

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_8330572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿúú::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
±
®
F__inference_dense_9716_layer_call_and_return_conditional_losses_833099

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 ±	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ±	:::Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	
 
_user_specified_nameinputs
í

K__inference_sequential_4858_layer_call_and_return_conditional_losses_833296

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource-
)dense_9716_matmul_readvariableop_resource.
*dense_9716_biasadd_readvariableop_resource-
)dense_9717_matmul_readvariableop_resource.
*dense_9717_biasadd_readvariableop_resource
identity°
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02 
conv2d_5/Conv2D/ReadVariableOpÁ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
*
paddingVALID*
strides
2
conv2d_5/Conv2D§
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp®
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
conv2d_5/BiasAdd}
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿøø
2
conv2d_5/ReluÇ
max_pooling2d_4/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||
*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ X 2
flatten_3/Const¡
flatten_3/ReshapeReshape max_pooling2d_4/MaxPool:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2
flatten_3/Reshape°
 dense_9716/MatMul/ReadVariableOpReadVariableOp)dense_9716_matmul_readvariableop_resource* 
_output_shapes
:
 ±	
*
dtype02"
 dense_9716/MatMul/ReadVariableOp¨
dense_9716/MatMulMatMulflatten_3/Reshape:output:0(dense_9716/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9716/MatMul­
!dense_9716/BiasAdd/ReadVariableOpReadVariableOp*dense_9716_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!dense_9716/BiasAdd/ReadVariableOp­
dense_9716/BiasAddBiasAdddense_9716/MatMul:product:0)dense_9716/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9716/BiasAddy
dense_9716/ReluReludense_9716/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_9716/Relu®
 dense_9717/MatMul/ReadVariableOpReadVariableOp)dense_9717_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02"
 dense_9717/MatMul/ReadVariableOp«
dense_9717/MatMulMatMuldense_9716/Relu:activations:0(dense_9717/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9717/MatMul­
!dense_9717/BiasAdd/ReadVariableOpReadVariableOp*dense_9717_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_9717/BiasAdd/ReadVariableOp­
dense_9717/BiasAddBiasAdddense_9717/MatMul:product:0)dense_9717/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9717/BiasAdd
dense_9717/SoftmaxSoftmaxdense_9717/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9717/Softmaxp
IdentityIdentitydense_9717/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú:::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
á
½
$__inference_signature_wrapper_833268
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_8330302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
(
_user_specified_nameconv2d_5_input
¿
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_833384

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ X 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ||
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ||

 
_user_specified_nameinputs
ÿ
Á
0__inference_sequential_4858_layer_call_fn_833341

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_4858_layer_call_and_return_conditional_losses_8331882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿúú::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
å

+__inference_dense_9716_layer_call_fn_833409

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_9716_layer_call_and_return_conditional_losses_8330992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ±	::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±	
 
_user_specified_nameinputs
­
L
0__inference_max_pooling2d_4_layer_call_fn_833042

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_8330362
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Å
serving_default±
S
conv2d_5_inputA
 serving_default_conv2d_5_input:0ÿÿÿÿÿÿÿÿÿúú>

dense_97170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ó¶
-
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
e__call__
f_default_save_signature
*g&call_and_return_all_conditional_losses"³*
_tf_keras_sequential*{"class_name": "Sequential", "name": "sequential_4858", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4858", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 250, 250, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9716", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9717", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 250, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4858", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 250, 250, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_9716", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9717", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}


_inbound_nodes

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 250, 3]}}

_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}

_inbound_nodes

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
n__call__
*o&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Dense", "name": "dense_9716", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9716", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 153760}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 153760]}}

$_inbound_nodes

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
p__call__
*q&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "dense_9717", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9717", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
¿
+iter

,beta_1

-beta_2
	.decay
/learning_ratemYmZm[m\%m]&m^v_v`vavb%vc&vd"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
Ê
0metrics

1layers
2non_trainable_variables
3layer_regularization_losses
regularization_losses
trainable_variables
		variables
4layer_metrics
e__call__
f_default_save_signature
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
,
rserving_default"
signature_map
 "
trackable_list_wrapper
):'
2conv2d_5/kernel
:
2conv2d_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
5metrics

6layers
7non_trainable_variables
8layer_regularization_losses
regularization_losses
trainable_variables
	variables
9layer_metrics
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
:metrics

;layers
<non_trainable_variables
=layer_regularization_losses
regularization_losses
trainable_variables
	variables
>layer_metrics
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?metrics

@layers
Anon_trainable_variables
Blayer_regularization_losses
regularization_losses
trainable_variables
	variables
Clayer_metrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
%:#
 ±	
2dense_9716/kernel
:
2dense_9716/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Dmetrics

Elayers
Fnon_trainable_variables
Glayer_regularization_losses
 regularization_losses
!trainable_variables
"	variables
Hlayer_metrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
#:!
2dense_9717/kernel
:2dense_9717/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
­
Imetrics

Jlayers
Knon_trainable_variables
Llayer_regularization_losses
'regularization_losses
(trainable_variables
)	variables
Mlayer_metrics
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
N0
O1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"¿
_tf_keras_metric¤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
.:,
2Adam/conv2d_5/kernel/m
 :
2Adam/conv2d_5/bias/m
*:(
 ±	
2Adam/dense_9716/kernel/m
": 
2Adam/dense_9716/bias/m
(:&
2Adam/dense_9717/kernel/m
": 2Adam/dense_9717/bias/m
.:,
2Adam/conv2d_5/kernel/v
 :
2Adam/conv2d_5/bias/v
*:(
 ±	
2Adam/dense_9716/kernel/v
": 
2Adam/dense_9716/bias/v
(:&
2Adam/dense_9717/kernel/v
": 2Adam/dense_9717/bias/v
2
0__inference_sequential_4858_layer_call_fn_833203
0__inference_sequential_4858_layer_call_fn_833358
0__inference_sequential_4858_layer_call_fn_833341
0__inference_sequential_4858_layer_call_fn_833241À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
!__inference__wrapped_model_833030Ç
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú
ú2÷
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833296
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833164
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833324
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833143À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_5_layer_call_fn_833378¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_5_layer_call_and_return_conditional_losses_833369¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_4_layer_call_fn_833042à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_833036à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_flatten_3_layer_call_fn_833389¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_3_layer_call_and_return_conditional_losses_833384¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_9716_layer_call_fn_833409¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_9716_layer_call_and_return_conditional_losses_833400¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_9717_layer_call_fn_833429¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_9717_layer_call_and_return_conditional_losses_833420¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:B8
$__inference_signature_wrapper_833268conv2d_5_inputª
!__inference__wrapped_model_833030%&A¢>
7¢4
2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú
ª "7ª4
2

dense_9717$!

dense_9717ÿÿÿÿÿÿÿÿÿ¸
D__inference_conv2d_5_layer_call_and_return_conditional_losses_833369p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿúú
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿøø

 
)__inference_conv2d_5_layer_call_fn_833378c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿúú
ª ""ÿÿÿÿÿÿÿÿÿøø
¨
F__inference_dense_9716_layer_call_and_return_conditional_losses_833400^1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ ±	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
+__inference_dense_9716_layer_call_fn_833409Q1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ ±	
ª "ÿÿÿÿÿÿÿÿÿ
¦
F__inference_dense_9717_layer_call_and_return_conditional_losses_833420\%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_9717_layer_call_fn_833429O%&/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ«
E__inference_flatten_3_layer_call_and_return_conditional_losses_833384b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ||

ª "'¢$

0ÿÿÿÿÿÿÿÿÿ ±	
 
*__inference_flatten_3_layer_call_fn_833389U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ||

ª "ÿÿÿÿÿÿÿÿÿ ±	î
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_833036R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_4_layer_call_fn_833042R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÉ
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833143z%&I¢F
?¢<
2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833164z%&I¢F
?¢<
2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833296r%&A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿúú
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_4858_layer_call_and_return_conditional_losses_833324r%&A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿúú
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
0__inference_sequential_4858_layer_call_fn_833203m%&I¢F
?¢<
2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú
p

 
ª "ÿÿÿÿÿÿÿÿÿ¡
0__inference_sequential_4858_layer_call_fn_833241m%&I¢F
?¢<
2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_4858_layer_call_fn_833341e%&A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿúú
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_4858_layer_call_fn_833358e%&A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿúú
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¿
$__inference_signature_wrapper_833268%&S¢P
¢ 
IªF
D
conv2d_5_input2/
conv2d_5_inputÿÿÿÿÿÿÿÿÿúú"7ª4
2

dense_9717$!

dense_9717ÿÿÿÿÿÿÿÿÿ