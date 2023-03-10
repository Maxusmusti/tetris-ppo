��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	� *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:  *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

: *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
|
training_2/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_2/Adam/iter
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
_output_shapes
: *
dtype0	
�
training_2/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_2/Adam/beta_1
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
_output_shapes
: *
dtype0
�
training_2/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_2/Adam/beta_2
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_2/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_2/Adam/decay
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
_output_shapes
: *
dtype0
�
training_2/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_2/Adam/learning_rate
�
1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
_output_shapes
: *
dtype0
�
!training_2/Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!training_2/Adam/conv2d_1/kernel/m
�
5training_2/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
�
training_2/Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_2/Adam/conv2d_1/bias/m
�
3training_2/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
�
 training_2/Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *1
shared_name" training_2/Adam/dense_3/kernel/m
�
4training_2/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_3/kernel/m*
_output_shapes
:	� *
dtype0
�
training_2/Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training_2/Adam/dense_3/bias/m
�
2training_2/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_3/bias/m*
_output_shapes
: *
dtype0
�
 training_2/Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *1
shared_name" training_2/Adam/dense_4/kernel/m
�
4training_2/Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_4/kernel/m*
_output_shapes

:  *
dtype0
�
training_2/Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training_2/Adam/dense_4/bias/m
�
2training_2/Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_4/bias/m*
_output_shapes
: *
dtype0
�
 training_2/Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" training_2/Adam/dense_5/kernel/m
�
4training_2/Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_5/kernel/m*
_output_shapes

: *
dtype0
�
training_2/Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training_2/Adam/dense_5/bias/m
�
2training_2/Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_5/bias/m*
_output_shapes
:*
dtype0
�
!training_2/Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!training_2/Adam/conv2d_1/kernel/v
�
5training_2/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!training_2/Adam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
�
training_2/Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_2/Adam/conv2d_1/bias/v
�
3training_2/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
�
 training_2/Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *1
shared_name" training_2/Adam/dense_3/kernel/v
�
4training_2/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_3/kernel/v*
_output_shapes
:	� *
dtype0
�
training_2/Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training_2/Adam/dense_3/bias/v
�
2training_2/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_3/bias/v*
_output_shapes
: *
dtype0
�
 training_2/Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *1
shared_name" training_2/Adam/dense_4/kernel/v
�
4training_2/Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_4/kernel/v*
_output_shapes

:  *
dtype0
�
training_2/Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training_2/Adam/dense_4/bias/v
�
2training_2/Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_4/bias/v*
_output_shapes
: *
dtype0
�
 training_2/Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" training_2/Adam/dense_5/kernel/v
�
4training_2/Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp training_2/Adam/dense_5/kernel/v*
_output_shapes

: *
dtype0
�
training_2/Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training_2/Adam/dense_5/bias/v
�
2training_2/Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOptraining_2/Adam/dense_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�4
value�4B�4 B�4
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
 
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�
9iter

:beta_1

;beta_2
	<decay
=learning_ratemkml'mm(mn-mo.mp3mq4mrvsvt'vu(vv-vw.vx3vy4vz
8
0
1
'2
(3
-4
.5
36
47
8
0
1
'2
(3
-4
.5
36
47
 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
 trainable_variables
!regularization_losses
 
 
 
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
#	variables
$trainable_variables
%regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
)	variables
*trainable_variables
+regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
/	variables
0trainable_variables
1regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
5	variables
6trainable_variables
7regularization_losses
SQ
VARIABLE_VALUEtraining_2/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_2/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_2/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
F
0
1
2
3
4
5
6
7
	8

9
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
��
VARIABLE_VALUE!training_2/Adam/conv2d_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/conv2d_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE training_2/Adam/dense_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/dense_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE training_2/Adam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE training_2/Adam/dense_5/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/dense_5/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!training_2/Adam/conv2d_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/conv2d_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE training_2/Adam/dense_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/dense_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE training_2/Adam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE training_2/Adam/dense_5/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining_2/Adam/dense_5/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_6Placeholder*0
_output_shapes
:����������P*
dtype0*%
shape:����������P
z
serving_default_input_7Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6serving_default_input_7conv2d_1/kernelconv2d_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_8773
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp(training_2/Adam/iter/Read/ReadVariableOp*training_2/Adam/beta_1/Read/ReadVariableOp*training_2/Adam/beta_2/Read/ReadVariableOp)training_2/Adam/decay/Read/ReadVariableOp1training_2/Adam/learning_rate/Read/ReadVariableOp5training_2/Adam/conv2d_1/kernel/m/Read/ReadVariableOp3training_2/Adam/conv2d_1/bias/m/Read/ReadVariableOp4training_2/Adam/dense_3/kernel/m/Read/ReadVariableOp2training_2/Adam/dense_3/bias/m/Read/ReadVariableOp4training_2/Adam/dense_4/kernel/m/Read/ReadVariableOp2training_2/Adam/dense_4/bias/m/Read/ReadVariableOp4training_2/Adam/dense_5/kernel/m/Read/ReadVariableOp2training_2/Adam/dense_5/bias/m/Read/ReadVariableOp5training_2/Adam/conv2d_1/kernel/v/Read/ReadVariableOp3training_2/Adam/conv2d_1/bias/v/Read/ReadVariableOp4training_2/Adam/dense_3/kernel/v/Read/ReadVariableOp2training_2/Adam/dense_3/bias/v/Read/ReadVariableOp4training_2/Adam/dense_4/kernel/v/Read/ReadVariableOp2training_2/Adam/dense_4/bias/v/Read/ReadVariableOp4training_2/Adam/dense_5/kernel/v/Read/ReadVariableOp2training_2/Adam/dense_5/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_9121
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biastraining_2/Adam/itertraining_2/Adam/beta_1training_2/Adam/beta_2training_2/Adam/decaytraining_2/Adam/learning_rate!training_2/Adam/conv2d_1/kernel/mtraining_2/Adam/conv2d_1/bias/m training_2/Adam/dense_3/kernel/mtraining_2/Adam/dense_3/bias/m training_2/Adam/dense_4/kernel/mtraining_2/Adam/dense_4/bias/m training_2/Adam/dense_5/kernel/mtraining_2/Adam/dense_5/bias/m!training_2/Adam/conv2d_1/kernel/vtraining_2/Adam/conv2d_1/bias/v training_2/Adam/dense_3/kernel/vtraining_2/Adam/dense_3/bias/v training_2/Adam/dense_4/kernel/vtraining_2/Adam/dense_4/bias/v training_2/Adam/dense_5/kernel/vtraining_2/Adam/dense_5/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_9218��
�-
�
A__inference_model_3_layer_call_and_return_conditional_losses_8841
inputs_0
inputs_1H
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel:;
-conv2d_1_biasadd_readvariableop_conv2d_1_bias:?
,dense_3_matmul_readvariableop_dense_3_kernel:	� 9
+dense_3_biasadd_readvariableop_dense_3_bias: >
,dense_4_matmul_readvariableop_dense_4_kernel:  9
+dense_4_biasadd_readvariableop_dense_4_bias: >
,dense_5_matmul_readvariableop_dense_5_kernel: 9
+dense_5_biasadd_readvariableop_dense_5_bias:
identity��conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
max_pooling2d_2/MaxPoolMaxPoolinputs_0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
flatten_3/ReshapeReshapeconv2d_1/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
flatten_2/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2flatten_3/Reshape:output:0flatten_2/Reshape:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp,dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	� *
dtype0�
dense_3/MatMulMatMulconcatenate_1/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
: *
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_4/MatMul/ReadVariableOpReadVariableOp,dense_4_matmul_readvariableop_dense_4_kernel*
_output_shapes

:  *
dtype0�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
: *
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_5/MatMul/ReadVariableOpReadVariableOp,dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes

: *
dtype0�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Z V
0
_output_shapes
:����������P
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8384

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������P:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�

�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8919

inputs?
%conv2d_readvariableop_conv2d_1_kernel:2
$biasadd_readvariableop_conv2d_1_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
A__inference_dense_3_layer_call_and_return_conditional_losses_8974

inputs7
$matmul_readvariableop_dense_3_kernel:	� 1
#biasadd_readvariableop_dense_3_bias: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_3_kernel*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�y
�
 __inference__traced_restore_9218
file_prefix:
 assignvariableop_conv2d_1_kernel:.
 assignvariableop_1_conv2d_1_bias:4
!assignvariableop_2_dense_3_kernel:	� -
assignvariableop_3_dense_3_bias: 3
!assignvariableop_4_dense_4_kernel:  -
assignvariableop_5_dense_4_bias: 3
!assignvariableop_6_dense_5_kernel: -
assignvariableop_7_dense_5_bias:1
'assignvariableop_8_training_2_adam_iter:	 3
)assignvariableop_9_training_2_adam_beta_1: 4
*assignvariableop_10_training_2_adam_beta_2: 3
)assignvariableop_11_training_2_adam_decay: ;
1assignvariableop_12_training_2_adam_learning_rate: O
5assignvariableop_13_training_2_adam_conv2d_1_kernel_m:A
3assignvariableop_14_training_2_adam_conv2d_1_bias_m:G
4assignvariableop_15_training_2_adam_dense_3_kernel_m:	� @
2assignvariableop_16_training_2_adam_dense_3_bias_m: F
4assignvariableop_17_training_2_adam_dense_4_kernel_m:  @
2assignvariableop_18_training_2_adam_dense_4_bias_m: F
4assignvariableop_19_training_2_adam_dense_5_kernel_m: @
2assignvariableop_20_training_2_adam_dense_5_bias_m:O
5assignvariableop_21_training_2_adam_conv2d_1_kernel_v:A
3assignvariableop_22_training_2_adam_conv2d_1_bias_v:G
4assignvariableop_23_training_2_adam_dense_3_kernel_v:	� @
2assignvariableop_24_training_2_adam_dense_3_bias_v: F
4assignvariableop_25_training_2_adam_dense_4_kernel_v:  @
2assignvariableop_26_training_2_adam_dense_4_bias_v: F
4assignvariableop_27_training_2_adam_dense_5_kernel_v: @
2assignvariableop_28_training_2_adam_dense_5_bias_v:
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_2_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_training_2_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp*assignvariableop_10_training_2_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_training_2_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp1assignvariableop_12_training_2_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp5assignvariableop_13_training_2_adam_conv2d_1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp3assignvariableop_14_training_2_adam_conv2d_1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp4assignvariableop_15_training_2_adam_dense_3_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp2assignvariableop_16_training_2_adam_dense_3_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp4assignvariableop_17_training_2_adam_dense_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp2assignvariableop_18_training_2_adam_dense_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_training_2_adam_dense_5_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp2assignvariableop_20_training_2_adam_dense_5_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_training_2_adam_conv2d_1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp3assignvariableop_22_training_2_adam_conv2d_1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp4assignvariableop_23_training_2_adam_dense_3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp2assignvariableop_24_training_2_adam_dense_3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp4assignvariableop_25_training_2_adam_dense_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp2assignvariableop_26_training_2_adam_dense_4_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_training_2_adam_dense_5_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp2assignvariableop_28_training_2_adam_dense_5_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
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
�
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_8407

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�D
�
__inference__traced_save_9121
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop3
/savev2_training_2_adam_iter_read_readvariableop	5
1savev2_training_2_adam_beta_1_read_readvariableop5
1savev2_training_2_adam_beta_2_read_readvariableop4
0savev2_training_2_adam_decay_read_readvariableop<
8savev2_training_2_adam_learning_rate_read_readvariableop@
<savev2_training_2_adam_conv2d_1_kernel_m_read_readvariableop>
:savev2_training_2_adam_conv2d_1_bias_m_read_readvariableop?
;savev2_training_2_adam_dense_3_kernel_m_read_readvariableop=
9savev2_training_2_adam_dense_3_bias_m_read_readvariableop?
;savev2_training_2_adam_dense_4_kernel_m_read_readvariableop=
9savev2_training_2_adam_dense_4_bias_m_read_readvariableop?
;savev2_training_2_adam_dense_5_kernel_m_read_readvariableop=
9savev2_training_2_adam_dense_5_bias_m_read_readvariableop@
<savev2_training_2_adam_conv2d_1_kernel_v_read_readvariableop>
:savev2_training_2_adam_conv2d_1_bias_v_read_readvariableop?
;savev2_training_2_adam_dense_3_kernel_v_read_readvariableop=
9savev2_training_2_adam_dense_3_bias_v_read_readvariableop?
;savev2_training_2_adam_dense_4_kernel_v_read_readvariableop=
9savev2_training_2_adam_dense_4_bias_v_read_readvariableop?
;savev2_training_2_adam_dense_5_kernel_v_read_readvariableop=
9savev2_training_2_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop/savev2_training_2_adam_iter_read_readvariableop1savev2_training_2_adam_beta_1_read_readvariableop1savev2_training_2_adam_beta_2_read_readvariableop0savev2_training_2_adam_decay_read_readvariableop8savev2_training_2_adam_learning_rate_read_readvariableop<savev2_training_2_adam_conv2d_1_kernel_m_read_readvariableop:savev2_training_2_adam_conv2d_1_bias_m_read_readvariableop;savev2_training_2_adam_dense_3_kernel_m_read_readvariableop9savev2_training_2_adam_dense_3_bias_m_read_readvariableop;savev2_training_2_adam_dense_4_kernel_m_read_readvariableop9savev2_training_2_adam_dense_4_bias_m_read_readvariableop;savev2_training_2_adam_dense_5_kernel_m_read_readvariableop9savev2_training_2_adam_dense_5_bias_m_read_readvariableop<savev2_training_2_adam_conv2d_1_kernel_v_read_readvariableop:savev2_training_2_adam_conv2d_1_bias_v_read_readvariableop;savev2_training_2_adam_dense_3_kernel_v_read_readvariableop9savev2_training_2_adam_dense_3_bias_v_read_readvariableop;savev2_training_2_adam_dense_4_kernel_v_read_readvariableop9savev2_training_2_adam_dense_4_bias_v_read_readvariableop;savev2_training_2_adam_dense_5_kernel_v_read_readvariableop9savev2_training_2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::	� : :  : : :: : : : : :::	� : :  : : ::::	� : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	
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
: :,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
�

�
&__inference_model_3_layer_call_fn_8715
input_6
input_7)
conv2d_1_kernel:
conv2d_1_bias:!
dense_3_kernel:	� 
dense_3_bias:  
dense_4_kernel:  
dense_4_bias:  
dense_5_kernel: 
dense_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6input_7conv2d_1_kernelconv2d_1_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_biasdense_5_kerneldense_5_bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_8648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_6:PL
'
_output_shapes
:���������
!
_user_specified_name	input_7
�
�
&__inference_dense_5_layer_call_fn_8999

inputs 
dense_5_kernel: 
dense_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_8468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8901

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������
"
identityIdentity:output:0*/
_input_shapes
:����������P:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_2_layer_call_fn_8891

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8384h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������
"
identityIdentity:output:0*/
_input_shapes
:����������P:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�$
�
A__inference_model_3_layer_call_and_return_conditional_losses_8648

inputs
inputs_12
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:)
dense_3_dense_3_kernel:	� "
dense_3_dense_3_bias: (
dense_4_dense_4_kernel:  "
dense_4_dense_4_bias: (
dense_5_dense_5_kernel: "
dense_5_dense_5_bias:
identity�� conv2d_1/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8384�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8397�
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8407�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8415�
concatenate_1/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8425�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_8438�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_8453�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_8468w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:����������P:���������: : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
A__inference_model_3_layer_call_and_return_conditional_losses_8473

inputs
inputs_12
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:)
dense_3_dense_3_kernel:	� "
dense_3_dense_3_bias: (
dense_4_dense_4_kernel:  "
dense_4_dense_4_bias: (
dense_5_dense_5_kernel: "
dense_5_dense_5_bias:
identity�� conv2d_1/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8384�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8397�
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8407�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8415�
concatenate_1/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8425�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_8438�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_8453�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_8468w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:����������P:���������: : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�-
�
A__inference_model_3_layer_call_and_return_conditional_losses_8881
inputs_0
inputs_1H
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel:;
-conv2d_1_biasadd_readvariableop_conv2d_1_bias:?
,dense_3_matmul_readvariableop_dense_3_kernel:	� 9
+dense_3_biasadd_readvariableop_dense_3_bias: >
,dense_4_matmul_readvariableop_dense_4_kernel:  9
+dense_4_biasadd_readvariableop_dense_4_bias: >
,dense_5_matmul_readvariableop_dense_5_kernel: 9
+dense_5_biasadd_readvariableop_dense_5_bias:
identity��conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
max_pooling2d_2/MaxPoolMaxPoolinputs_0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
flatten_3/ReshapeReshapeconv2d_1/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
flatten_2/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2flatten_3/Reshape:output:0flatten_2/Reshape:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp,dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	� *
dtype0�
dense_3/MatMulMatMulconcatenate_1/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
: *
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_4/MatMul/ReadVariableOpReadVariableOp,dense_4_matmul_readvariableop_dense_4_kernel*
_output_shapes

:  *
dtype0�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
: *
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_5/MatMul/ReadVariableOpReadVariableOp,dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes

: *
dtype0�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Z V
0
_output_shapes
:����������P
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
A__inference_dense_4_layer_call_and_return_conditional_losses_8992

inputs6
$matmul_readvariableop_dense_4_kernel:  1
#biasadd_readvariableop_dense_4_bias: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_4_kernel*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�#
�
A__inference_model_3_layer_call_and_return_conditional_losses_8736
input_6
input_72
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:)
dense_3_dense_3_kernel:	� "
dense_3_dense_3_bias: (
dense_4_dense_4_kernel:  "
dense_4_dense_4_bias: (
dense_5_dense_5_kernel: "
dense_5_dense_5_bias:
identity�� conv2d_1/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8384�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8397�
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8407�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8415�
concatenate_1/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0input_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8425�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_8438�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_8453�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_8468w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_6:PL
'
_output_shapes
:���������
!
_user_specified_name	input_7
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8896

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_8415

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�

G__inference_concatenate_1_layer_call_and_return_conditional_losses_8425

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:����������:����������:���������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_conv2d_1_layer_call_fn_8908

inputs)
conv2d_1_kernel:
conv2d_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernelconv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8397w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
A__inference_dense_3_layer_call_and_return_conditional_losses_8438

inputs7
$matmul_readvariableop_dense_3_kernel:	� 1
#biasadd_readvariableop_dense_3_bias: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_3_kernel*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_5_layer_call_and_return_conditional_losses_8468

inputs6
$matmul_readvariableop_dense_5_kernel: 1
#biasadd_readvariableop_dense_5_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_5_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8363

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8956
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*N
_input_shapes=
;:����������:����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
D
(__inference_flatten_3_layer_call_fn_8924

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8407a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
&__inference_model_3_layer_call_fn_8801
inputs_0
inputs_1)
conv2d_1_kernel:
conv2d_1_bias:!
dense_3_kernel:	� 
dense_3_bias:  
dense_4_kernel:  
dense_4_bias:  
dense_5_kernel: 
dense_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1conv2d_1_kernelconv2d_1_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_biasdense_5_kerneldense_5_bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_8648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������P
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
&__inference_dense_3_layer_call_fn_8963

inputs!
dense_3_kernel:	� 
dense_3_bias: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_kerneldense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_8438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�3
�
__inference__wrapped_model_8354
input_6
input_7P
6model_3_conv2d_1_conv2d_readvariableop_conv2d_1_kernel:C
5model_3_conv2d_1_biasadd_readvariableop_conv2d_1_bias:G
4model_3_dense_3_matmul_readvariableop_dense_3_kernel:	� A
3model_3_dense_3_biasadd_readvariableop_dense_3_bias: F
4model_3_dense_4_matmul_readvariableop_dense_4_kernel:  A
3model_3_dense_4_biasadd_readvariableop_dense_4_bias: F
4model_3_dense_5_matmul_readvariableop_dense_5_kernel: A
3model_3_dense_5_biasadd_readvariableop_dense_5_bias:
identity��'model_3/conv2d_1/BiasAdd/ReadVariableOp�&model_3/conv2d_1/Conv2D/ReadVariableOp�&model_3/dense_3/BiasAdd/ReadVariableOp�%model_3/dense_3/MatMul/ReadVariableOp�&model_3/dense_4/BiasAdd/ReadVariableOp�%model_3/dense_4/MatMul/ReadVariableOp�&model_3/dense_5/BiasAdd/ReadVariableOp�%model_3/dense_5/MatMul/ReadVariableOp�
model_3/max_pooling2d_2/MaxPoolMaxPoolinput_6*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
�
&model_3/conv2d_1/Conv2D/ReadVariableOpReadVariableOp6model_3_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
model_3/conv2d_1/Conv2DConv2D(model_3/max_pooling2d_2/MaxPool:output:0.model_3/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
'model_3/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5model_3_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0�
model_3/conv2d_1/BiasAddBiasAdd model_3/conv2d_1/Conv2D:output:0/model_3/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������z
model_3/conv2d_1/ReluRelu!model_3/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������h
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
model_3/flatten_3/ReshapeReshape#model_3/conv2d_1/Relu:activations:0 model_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������h
model_3/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
model_3/flatten_2/ReshapeReshape(model_3/max_pooling2d_2/MaxPool:output:0 model_3/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������c
!model_3/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_3/concatenate_1/concatConcatV2"model_3/flatten_3/Reshape:output:0"model_3/flatten_2/Reshape:output:0input_7*model_3/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
%model_3/dense_3/MatMul/ReadVariableOpReadVariableOp4model_3_dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	� *
dtype0�
model_3/dense_3/MatMulMatMul%model_3/concatenate_1/concat:output:0-model_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp3model_3_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
: *
dtype0�
model_3/dense_3/BiasAddBiasAdd model_3/dense_3/MatMul:product:0.model_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
model_3/dense_3/ReluRelu model_3/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
%model_3/dense_4/MatMul/ReadVariableOpReadVariableOp4model_3_dense_4_matmul_readvariableop_dense_4_kernel*
_output_shapes

:  *
dtype0�
model_3/dense_4/MatMulMatMul"model_3/dense_3/Relu:activations:0-model_3/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp3model_3_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
: *
dtype0�
model_3/dense_4/BiasAddBiasAdd model_3/dense_4/MatMul:product:0.model_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
model_3/dense_4/ReluRelu model_3/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
%model_3/dense_5/MatMul/ReadVariableOpReadVariableOp4model_3_dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes

: *
dtype0�
model_3/dense_5/MatMulMatMul"model_3/dense_4/Relu:activations:0-model_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp3model_3_dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0�
model_3/dense_5/BiasAddBiasAdd model_3/dense_5/MatMul:product:0.model_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
model_3/dense_5/SigmoidSigmoid model_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitymodel_3/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_3/conv2d_1/BiasAdd/ReadVariableOp'^model_3/conv2d_1/Conv2D/ReadVariableOp'^model_3/dense_3/BiasAdd/ReadVariableOp&^model_3/dense_3/MatMul/ReadVariableOp'^model_3/dense_4/BiasAdd/ReadVariableOp&^model_3/dense_4/MatMul/ReadVariableOp'^model_3/dense_5/BiasAdd/ReadVariableOp&^model_3/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 2R
'model_3/conv2d_1/BiasAdd/ReadVariableOp'model_3/conv2d_1/BiasAdd/ReadVariableOp2P
&model_3/conv2d_1/Conv2D/ReadVariableOp&model_3/conv2d_1/Conv2D/ReadVariableOp2P
&model_3/dense_3/BiasAdd/ReadVariableOp&model_3/dense_3/BiasAdd/ReadVariableOp2N
%model_3/dense_3/MatMul/ReadVariableOp%model_3/dense_3/MatMul/ReadVariableOp2P
&model_3/dense_4/BiasAdd/ReadVariableOp&model_3/dense_4/BiasAdd/ReadVariableOp2N
%model_3/dense_4/MatMul/ReadVariableOp%model_3/dense_4/MatMul/ReadVariableOp2P
&model_3/dense_5/BiasAdd/ReadVariableOp&model_3/dense_5/BiasAdd/ReadVariableOp2N
%model_3/dense_5/MatMul/ReadVariableOp%model_3/dense_5/MatMul/ReadVariableOp:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_6:PL
'
_output_shapes
:���������
!
_user_specified_name	input_7
�
f
,__inference_concatenate_1_layer_call_fn_8948
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8425a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*N
_input_shapes=
;:����������:����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�

�
&__inference_model_3_layer_call_fn_8787
inputs_0
inputs_1)
conv2d_1_kernel:
conv2d_1_bias:!
dense_3_kernel:	� 
dense_3_bias:  
dense_4_kernel:  
dense_4_bias:  
dense_5_kernel: 
dense_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1conv2d_1_kernelconv2d_1_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_biasdense_5_kerneldense_5_bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_8473o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������P
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8397

inputs?
%conv2d_readvariableop_conv2d_1_kernel:2
$biasadd_readvariableop_conv2d_1_bias:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
&__inference_model_3_layer_call_fn_8484
input_6
input_7)
conv2d_1_kernel:
conv2d_1_bias:!
dense_3_kernel:	� 
dense_3_bias:  
dense_4_kernel:  
dense_4_bias:  
dense_5_kernel: 
dense_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6input_7conv2d_1_kernelconv2d_1_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_biasdense_5_kerneldense_5_bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_3_layer_call_and_return_conditional_losses_8473o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_6:PL
'
_output_shapes
:���������
!
_user_specified_name	input_7
�
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_8930

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_8941

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�#
�
A__inference_model_3_layer_call_and_return_conditional_losses_8757
input_6
input_72
conv2d_1_conv2d_1_kernel:$
conv2d_1_conv2d_1_bias:)
dense_3_dense_3_kernel:	� "
dense_3_dense_3_bias: (
dense_4_dense_4_kernel:  "
dense_4_dense_4_bias: (
dense_5_dense_5_kernel: "
dense_5_dense_5_bias:
identity�� conv2d_1/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8384�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_1_conv2d_1_kernelconv2d_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8397�
flatten_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_8407�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8415�
concatenate_1/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0input_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8425�
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_8438�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_8453�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_8468w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^conv2d_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_6:PL
'
_output_shapes
:���������
!
_user_specified_name	input_7
�
�
&__inference_dense_4_layer_call_fn_8981

inputs 
dense_4_kernel:  
dense_4_bias: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_8453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
A__inference_dense_4_layer_call_and_return_conditional_losses_8453

inputs6
$matmul_readvariableop_dense_4_kernel:  1
#biasadd_readvariableop_dense_4_bias: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_4_kernel*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
"__inference_signature_wrapper_8773
input_6
input_7)
conv2d_1_kernel:
conv2d_1_bias:!
dense_3_kernel:	� 
dense_3_bias:  
dense_4_kernel:  
dense_4_bias:  
dense_5_kernel: 
dense_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6input_7conv2d_1_kernelconv2d_1_biasdense_3_kerneldense_3_biasdense_4_kerneldense_4_biasdense_5_kerneldense_5_bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_8354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*R
_input_shapesA
?:����������P:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_6:PL
'
_output_shapes
:���������
!
_user_specified_name	input_7
�
D
(__inference_flatten_2_layer_call_fn_8935

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_8415a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
A__inference_dense_5_layer_call_and_return_conditional_losses_9010

inputs6
$matmul_readvariableop_dense_5_kernel: 1
#biasadd_readvariableop_dense_5_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_5_kernel*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_2_layer_call_fn_8886

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8363�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input_69
serving_default_input_6:0����������P
;
input_70
serving_default_input_7:0���������;
dense_50
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
{__call__
*|&call_and_return_all_conditional_losses
}_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9iter

:beta_1

;beta_2
	<decay
=learning_ratemkml'mm(mn-mo.mp3mq4mrvsvt'vu(vv-vw.vx3vy4vz"
	optimizer
X
0
1
'2
(3
-4
.5
36
47"
trackable_list_wrapper
X
0
1
'2
(3
-4
.5
36
47"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
{__call__
}_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
 trainable_variables
!regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
#	variables
$trainable_variables
%regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	� 2dense_3/kernel
: 2dense_3/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
)	variables
*trainable_variables
+regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :  2dense_4/kernel
: 2dense_4/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
/	variables
0trainable_variables
1regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_5/kernel
:2dense_5/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
5	variables
6trainable_variables
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_2/Adam/iter
 : (2training_2/Adam/beta_1
 : (2training_2/Adam/beta_2
: (2training_2/Adam/decay
':% (2training_2/Adam/learning_rate
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
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
9:72!training_2/Adam/conv2d_1/kernel/m
+:)2training_2/Adam/conv2d_1/bias/m
1:/	� 2 training_2/Adam/dense_3/kernel/m
*:( 2training_2/Adam/dense_3/bias/m
0:.  2 training_2/Adam/dense_4/kernel/m
*:( 2training_2/Adam/dense_4/bias/m
0:. 2 training_2/Adam/dense_5/kernel/m
*:(2training_2/Adam/dense_5/bias/m
9:72!training_2/Adam/conv2d_1/kernel/v
+:)2training_2/Adam/conv2d_1/bias/v
1:/	� 2 training_2/Adam/dense_3/kernel/v
*:( 2training_2/Adam/dense_3/bias/v
0:.  2 training_2/Adam/dense_4/kernel/v
*:( 2training_2/Adam/dense_4/bias/v
0:. 2 training_2/Adam/dense_5/kernel/v
*:(2training_2/Adam/dense_5/bias/v
�2�
&__inference_model_3_layer_call_fn_8484
&__inference_model_3_layer_call_fn_8787
&__inference_model_3_layer_call_fn_8801
&__inference_model_3_layer_call_fn_8715�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_3_layer_call_and_return_conditional_losses_8841
A__inference_model_3_layer_call_and_return_conditional_losses_8881
A__inference_model_3_layer_call_and_return_conditional_losses_8736
A__inference_model_3_layer_call_and_return_conditional_losses_8757�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
__inference__wrapped_model_8354input_6input_7"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_max_pooling2d_2_layer_call_fn_8886
.__inference_max_pooling2d_2_layer_call_fn_8891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8896
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8901�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_1_layer_call_fn_8908�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8919�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_3_layer_call_fn_8924�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_flatten_3_layer_call_and_return_conditional_losses_8930�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_2_layer_call_fn_8935�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_flatten_2_layer_call_and_return_conditional_losses_8941�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_concatenate_1_layer_call_fn_8948�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8956�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_3_layer_call_fn_8963�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_3_layer_call_and_return_conditional_losses_8974�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_4_layer_call_fn_8981�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_4_layer_call_and_return_conditional_losses_8992�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_5_layer_call_fn_8999�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_5_layer_call_and_return_conditional_losses_9010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_8773input_6input_7"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_8354�'(-.34a�^
W�T
R�O
*�'
input_6����������P
!�
input_7���������
� "1�.
,
dense_5!�
dense_5����������
G__inference_concatenate_1_layer_call_and_return_conditional_losses_8956���}
v�s
q�n
#� 
inputs/0����������
#� 
inputs/1����������
"�
inputs/2���������
� "&�#
�
0����������
� �
,__inference_concatenate_1_layer_call_fn_8948���}
v�s
q�n
#� 
inputs/0����������
#� 
inputs/1����������
"�
inputs/2���������
� "������������
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8919l7�4
-�*
(�%
inputs���������

� "-�*
#� 
0���������
� �
'__inference_conv2d_1_layer_call_fn_8908_7�4
-�*
(�%
inputs���������

� " �����������
A__inference_dense_3_layer_call_and_return_conditional_losses_8974]'(0�-
&�#
!�
inputs����������
� "%�"
�
0��������� 
� z
&__inference_dense_3_layer_call_fn_8963P'(0�-
&�#
!�
inputs����������
� "���������� �
A__inference_dense_4_layer_call_and_return_conditional_losses_8992\-./�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� y
&__inference_dense_4_layer_call_fn_8981O-./�,
%�"
 �
inputs��������� 
� "���������� �
A__inference_dense_5_layer_call_and_return_conditional_losses_9010\34/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� y
&__inference_dense_5_layer_call_fn_8999O34/�,
%�"
 �
inputs��������� 
� "�����������
C__inference_flatten_2_layer_call_and_return_conditional_losses_8941a7�4
-�*
(�%
inputs���������

� "&�#
�
0����������
� �
(__inference_flatten_2_layer_call_fn_8935T7�4
-�*
(�%
inputs���������

� "������������
C__inference_flatten_3_layer_call_and_return_conditional_losses_8930a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_3_layer_call_fn_8924T7�4
-�*
(�%
inputs���������
� "������������
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8896�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_8901i8�5
.�+
)�&
inputs����������P
� "-�*
#� 
0���������

� �
.__inference_max_pooling2d_2_layer_call_fn_8886�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
.__inference_max_pooling2d_2_layer_call_fn_8891\8�5
.�+
)�&
inputs����������P
� " ����������
�
A__inference_model_3_layer_call_and_return_conditional_losses_8736�'(-.34i�f
_�\
R�O
*�'
input_6����������P
!�
input_7���������
p 

 
� "%�"
�
0���������
� �
A__inference_model_3_layer_call_and_return_conditional_losses_8757�'(-.34i�f
_�\
R�O
*�'
input_6����������P
!�
input_7���������
p

 
� "%�"
�
0���������
� �
A__inference_model_3_layer_call_and_return_conditional_losses_8841�'(-.34k�h
a�^
T�Q
+�(
inputs/0����������P
"�
inputs/1���������
p 

 
� "%�"
�
0���������
� �
A__inference_model_3_layer_call_and_return_conditional_losses_8881�'(-.34k�h
a�^
T�Q
+�(
inputs/0����������P
"�
inputs/1���������
p

 
� "%�"
�
0���������
� �
&__inference_model_3_layer_call_fn_8484�'(-.34i�f
_�\
R�O
*�'
input_6����������P
!�
input_7���������
p 

 
� "�����������
&__inference_model_3_layer_call_fn_8715�'(-.34i�f
_�\
R�O
*�'
input_6����������P
!�
input_7���������
p

 
� "�����������
&__inference_model_3_layer_call_fn_8787�'(-.34k�h
a�^
T�Q
+�(
inputs/0����������P
"�
inputs/1���������
p 

 
� "�����������
&__inference_model_3_layer_call_fn_8801�'(-.34k�h
a�^
T�Q
+�(
inputs/0����������P
"�
inputs/1���������
p

 
� "�����������
"__inference_signature_wrapper_8773�'(-.34r�o
� 
h�e
5
input_6*�'
input_6����������P
,
input_7!�
input_7���������"1�.
,
dense_5!�
dense_5���������