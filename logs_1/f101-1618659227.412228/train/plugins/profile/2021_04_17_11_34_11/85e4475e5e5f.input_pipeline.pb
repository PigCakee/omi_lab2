	kf-?'x@kf-?'x@!kf-?'x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-kf-?'x@o???? @1C??Cnw@A??'?bd??I?*P??c	@*#??~Z??@=
ף?M?@2?
XIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2@_?R#??9@!???:zQ@)_?R#??9@1???:zQ@:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecord@B?D|%@!?&̐?N=@)B?D|%@1?&̐?N=@:Advanced file read2?
aIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap@(??&2?%@!??Dn?=@)]?,σ???1H?O?/??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch????OV??!?tQ????)????OV??1?tQ????:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch??XP???!>{?P$r??)??XP???1>{?P$r??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTakeh?$????!O(?@????)?n?l???1??_???:Preprocessing2F
Iterator::Model???U???!?Ą????)???f???1?.??J??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismJ??%?L??!?^i?'???)b?c??1
?Y?B???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??U???@QrS?"@X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o???? @o???? @!o???? @      ??!       "	C??Cnw@C??Cnw@!C??Cnw@*      ??!       2	??'?bd????'?bd??!??'?bd??:	?*P??c	@?*P??c	@!?*P??c	@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??U???@yrS?"@X@