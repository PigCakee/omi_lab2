	$????RT@$????RT@!$????RT@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-$????RT@?#EdX?@1???\?(O@A?w~Q????I??^D?1+@*???K?s?@???K7?p@)      p=2?
_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::FlatMap[0]::TFRecord@Z?X"?!@!,9s??CX@)Z?X"?!@1,9s??CX@:Advanced file read2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::FlatMap@?9>Z?)"@!?DI?ĨX@)og_y????1%Ղu?G??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetchr??9???!g?GW7???)r??9???1g?GW7???:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch??????!o?b?????)??????1o?b?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?\?&???!]>??=??)L3?뤾??1."'?h???:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTakeG ^?/ح?!E??ԈB??)?/K;5???1#i?Rں??:Preprocessing2F
Iterator::Model?n???!<7?]??)7?X?O??1z??t麵?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?16.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI$\m??W7@Q????*S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#EdX?@?#EdX?@!?#EdX?@      ??!       "	???\?(O@???\?(O@!???\?(O@*      ??!       2	?w~Q?????w~Q????!?w~Q????:	??^D?1+@??^D?1+@!??^D?1+@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q$\m??W7@y????*S@