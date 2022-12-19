(*estimating 0.1*0.1 3h precipitation from 0.5*0.5 hourly circulation data*)

(*load the data*)
{p,d,plat,plon,dlat,dlon,ele}=Block[{tempt=Import["/data01/users/xzn/BX/Diffusion/EastAsia_data.mx"]},Map[tempt[#]&,{"p","d","plat","plon","dlat","dlon","ele"}]];
p=Map[NumericArray[Log[Normal[#]+1.],"Real16"]&,p];
{plat,plon,dlat,dlon}=Map[Floor[#,0.001]&,{plat,plon,dlat,dlon}];
test=4;

(*hyper-parameters*)
dRange=17;
size={80,80};
channel=1;
csize={17,17};
cchannel=15;
c=64;
depth=4;
{\[Lambda]min,\[Lambda]max}={-20,20};
\[Lambda]latent=256;
\[Beta]=Sort[Round[RandomReal[NormalDistribution[0,1],\[Lambda]latent/2],0.01]];
(*keey the beta*)
\[Beta]={-2.73`,-2.35`,-2.32`,-2.2`,-2.07`,-2.0100000000000002`,-1.8`,-1.75`,-1.75`,-1.72`,-1.6600000000000001`,-1.52`,-1.48`,-1.42`,-1.3`,-1.25`,
-1.25`,-1.24`,-1.24`,-1.19`,-1.1500000000000001`,-1.12`,-1.12`,-1.1`,-1.09`,-0.87`,-0.84`,-0.8`,-0.79`,-0.75`,-0.73`,-0.6900000000000001`,
-0.63`,-0.61`,-0.6`,-0.59`,-0.5700000000000001`,-0.53`,-0.51`,-0.5`,-0.5`,-0.49`,-0.48`,-0.44`,-0.37`,-0.3`,-0.29`,-0.27`,-0.27`,-0.25`,-0.23`,
-0.22`,-0.17`,-0.11`,-0.1`,-0.1`,-0.09`,-0.04`,-0.02`,0.`,0.01`,0.02`,0.06`,0.09`,0.13`,0.16`,0.18`,0.18`,0.22`,0.22`,0.24`,0.26`,0.28`,0.28`,
0.29`,0.32`,0.32`,0.33`,0.33`,0.36`,0.44`,0.47000000000000003`,0.6`,0.6`,0.63`,0.64`,0.67`,0.68`,0.72`,0.74`,0.77`,0.79`,0.8300000000000001`,
0.88`,0.9`,0.9`,0.9`,0.92`,0.9400000000000001`,0.9500000000000001`,0.97`,1.`,1.02`,1.07`,1.11`,1.1400000000000001`,1.16`,1.19`,1.22`,1.22`,1.3`,
1.33`,1.36`,1.3900000000000001`,1.4000000000000001`,1.45`,1.59`,1.6`,1.62`,1.69`,1.69`,1.72`,1.9100000000000001`,1.93`,2.04`,2.23`,2.31`,2.85`};
\[Lambda]Encoding=Flatten[{Cos[2Pi*\[Beta]*#],Sin[2Pi*\[Beta]*#]}]&;
\[Lambda][u_]:=Block[{a,b},b=ArcTan[Exp[-\[Lambda]max/2.]];a=ArcTan[Exp[-\[Lambda]min/2.]]-b; -2*Log[Tan[a*u+b]]]
\[Alpha][\[Lambda]_]:=Sqrt[1./(1+Exp[-\[Lambda]])]
\[Sigma][\[Lambda]_]:=Sqrt[1-\[Alpha][\[Lambda]]^2]
batch=64;

(*forward*)
Fzx[\[Lambda]_,x_]:=\[Alpha][\[Lambda]]*x+RandomReal[NormalDistribution[0,\[Sigma][\[Lambda]]],Dimensions[x]]
Fzx[\[Lambda]_,x_,\[Epsilon]_]:=\[Alpha][\[Lambda]]*x+\[Sigma][\[Lambda]]*\[Epsilon]

(*model blocks*)
res[c_]:=NetGraph[<|"long"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],Ramp},2]][[1;;-2]],
      "plus"->TotalLayer[],
      "short"->ConvolutionLayer[c,{1,1}]|>,
 {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

upres[c_,size_]:=NetGraph[<|"long"->{NormalizationLayer[],Ramp,ResizeLayer[size],ConvolutionLayer[c,{3,3},"PaddingSize"->1],                                
 NormalizationLayer[],Ramp,ConvolutionLayer[c,{3,3},"PaddingSize"->1]},
      "plus"->TotalLayer[],
      "short"->{ResizeLayer[size],ConvolutionLayer[c,{1,1}]}|>,
 {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

contract[channel_,crop_:{{1,1},{1,1}}]:=NetGraph[{"conv"->res[channel],"pooling"->PoolingLayer[2,2,"Function"->Mean],
                               "cropping"->PartLayer[{;;,crop[[1,1]];;-crop[[1,-1]],crop[[2,1]];;-crop[[2,-1]]}]},
                     {NetPort["Input"]->"conv"->"pooling"->NetPort["Pooling"],"conv"->"cropping"->NetPort["Shortcut"]}];

expand[channel_,size_]:=NetGraph[{"deconv"->upres[channel,size],
                        "join"->CatenateLayer[],
                        "conv"->res[channel/2]},
                     {NetPort["Input"]->"deconv"->"join",
                      NetPort["Shortcut"]->"join"->"conv"}];             
        
(*downscaling module*)
DownscalingNet=NetInitialize[NetGraph[Flatten[
  {Table["condition_"<>ToString[i]->{res[c],ElementwiseLayer["GELU"],upres[c*2^(i-1),{20,20}/2^(i-2)]},{i,0,4}],
  "preprocess"->{ConvolutionLayer[c,{3,3,3}],NormalizationLayer[],ElementwiseLayer["GELU"],
                ConvolutionLayer[c,{3,3,3}],NormalizationLayer[],ElementwiseLayer["GELU"],FlattenLayer[1]},
  "thread"->CatenateLayer[],
  Table["resize_"<>ToString[i]->ResizeLayer[{5*2^(4-i),5*2^(4-i)}],{i,0,4}],
  Table["mse_"<>ToString[i]->MeanSquaredLossLayer[],{i,0,4}],
  Table["postprocess_"<>ToString[i]->{ConvolutionLayer[c,{1,1}],NormalizationLayer[],ElementwiseLayer["GELU"],ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp},{i,0,4}],
  "EleProcess"->Append[Table[{res[c],ElementwiseLayer["GELU"]},{3}],{ConvolutionLayer[4,{5,5},"Stride"->5,"PaddingSize"->3],ReplicateLayer[9,2]}]}],
 Flatten[Join[Map["condition_"<>ToString[#[[1]]]->"condition_"<>ToString[#[[2]]]&,{{2,1},{1,0},{2,3},{3,4}}],
 {NetPort["Dynamics"]->"thread"->"preprocess"->"condition_2",
 NetPort["Elevation"]->"EleProcess"->"thread",
 Table[{"condition_"<>ToString[i]->"postprocess_"<>ToString[i]->"mse_"<>ToString[i],NetPort["P"]->"resize_"<>ToString[i]->"mse_"<>ToString[i]->NetPort["Loss_"<>ToString[i]]},{i,0,4}]}]],
 "Dynamics"->{15,9,17,17},
 "Elevation"->{1,80,80},
 "P"->{1,80,80}]];
 
(*train the downscaling module*)
validation=Block[{validation=Import["/data01/users/xzn/BX/Diffusion/EastAsia_Validation_Data.mx"]},
 Table[Block[{tempt=validation[[year,region]]},
   Map[<|"P"->(#[["NoiseP"]]-#[["Noise"]]*Sqrt[#[["scale"]]])/Sqrt[1-#[["scale"]]],
       "Dynamics"->#[["Dynamics"]],
       "Elevation"->#[["Elevation"]]|>&,tempt]],{year,Length[validation]},{region,Dimensions[validation][[2]]}]];

GlobeLoss=Infinity;
Report[net_,ValidationLoss_]:=Block[{},
        Print[{GlobeLoss,ValidationLoss}];
        If[ValidationLoss<GlobeLoss,Block[{},Print["Update"];
        Set[GlobeLoss,ValidationLoss];
        Export["/data01/users/xzn/BX/Diffusion/Downscaling_Net_12_07.mx",net]]]];

DownscalingNet=Import["/data01/users/xzn/BX/Diffusion/Downscaling_Net_12_07.mx"];

trained=NetTrain[DownscalingNet,
 {Function[Block[{u,\[Lambda]s,noise,select,precipitation,elevation,dynamics,position,\[Lambda]embedding,nprecipitation,condition,scale},
  select=Table[Block[{dPosition,dselect,pselect,pPosition},
        dPosition=Block[{tempt={RandomChoice[Range[1,Length[dlat]-dRange]],RandomChoice[Range[1,Length[dlon]-dRange]]}},
           {{tempt[[1]],tempt[[1]]+dRange-1},{tempt[[2]],tempt[[2]]+dRange-1}}];
        pPosition=(dPosition-1)*5+{{1,0},{1,0}};
       {Block[{year=RandomInteger[{1,Length[p]-test}]},{year,RandomInteger[{2,Length[p[[year]]]-1}]}],dPosition,pPosition}],batch];
  (*precipitation=Table[{Log[Normal[(p[[select[[i,1,1]]]][[select[[i,1,2]],select[[i,3,1,1]];;select[[i,3,1,2]],select[[i,3,2,1]];;select[[i,3,2,2]]]])]+1.]},{i,batch}];*)
  precipitation=Table[{Normal[(p[[select[[i,1,1]]]][[select[[i,1,2]],select[[i,3,1,1]];;select[[i,3,1,2]],select[[i,3,2,1]];;select[[i,3,2,2]]]])]},{i,batch}];
  elevation=Table[ele[[;;,select[[i,3,1,1]];;select[[i,3,1,2]],select[[i,3,2,1]];;select[[i,3,2,2]]]],{i,batch}];
  dynamics=Table[Transpose[d[[select[[i,1,1]]]][[(select[[i,1,2]]-2)*3+1;;(select[[i,1,2]]+1)*3,;;,select[[i,2,1,1]];;select[[i,2,1,2]],select[[i,2,2,1]];;select[[i,2,2,2]]]]],{i,batch}];
  <|"P"->precipitation,
   "Dynamics"->dynamics,
   "Elevation"->elevation|>]],"RoundLength"->batch*1000},
 LossFunction ->{"Loss_0"->Scaled[1],"Loss_1"->Scaled[.1],"Loss_2"->Scaled[.1],"Loss_3"->Scaled[.1],"Loss_4"->Scaled[.1]},
 BatchSize -> batch,
 MaxTrainingRounds->10000,
 TargetDevice->{"GPU",All},
 ValidationSet->Flatten[validation],
 Method->{"ADAM","LearningRate"->10^-3,"L2Regularization"->0},
 TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];

DownscalingNet=NetTake[DownscalingNet,"postprocess_0"];


(*unconditional diffusion model*)
UUNet=NetInitialize[NetGraph[<|Table[{"contract_"<>ToString[i]->contract[c*2^(i-1)],
                      "expand_"<>ToString[i]->expand[c*2^Max[(i-1),1],size/2^(i-1)],
                      "\[Lambda]_F_"<>ToString[i]->{LinearLayer[c*2^(i-1)],ReplicateLayer[Floor[size/2^i],2]},
                      "\[Lambda]_B_"<>ToString[i]->{LinearLayer[c*2^(i-1)],ReplicateLayer[Floor[size/2^i],2]},
                      "thread_F_"<>ToString[i]->ThreadingLayer[Plus],
                      "thread_B_"<>ToString[i]->ThreadingLayer[Plus]},{i,depth}],
                "\[Lambda]_F_0"->{LinearLayer[c],ReplicateLayer[size,2]},"thread_F_0"->ThreadingLayer[Plus],
                "\[Lambda]_B_0"->{LinearLayer[c],ReplicateLayer[size,2]},"thread_B_0"->ThreadingLayer[Plus],
                "preprocess"->ConvolutionLayer[c,{1,1}],
                "postprocess"->{ConvolutionLayer[c,{1,1}],NormalizationLayer[],ElementwiseLayer["GELU"],ConvolutionLayer[channel,{1,1}]},
                "ubase"->res[c*2^(depth-1)],
                "\[Lambda]Process"->{LinearLayer[2\[Lambda]latent],ElementwiseLayer["GELU"],LinearLayer[3\[Lambda]latent],ElementwiseLayer["GELU"]},
                "Loss"->MeanSquaredLossLayer[],
                "Rescale"->ThreadingLayer[Times]|>,
 Flatten[{
 NetPort["NoiseP"]->"preprocess"->"thread_F_0"->"contract_1","preprocess"->"thread_B_0",
 NetPort["\[Lambda]"]->"\[Lambda]Process"->"\[Lambda]_F_0"->"thread_F_0","\[Lambda]Process"->"\[Lambda]_B_0"->"thread_B_0",
 Table[{NetPort["contract_"<>ToString[i],"Pooling"]->"thread_F_"<>ToString[i]->
 If[i<depth,"contract_"<>ToString[i+1],"ubase"->"thread_B_"<>ToString[depth]->NetPort["expand_"<>ToString[depth],"Input"]],
 "\[Lambda]Process"->"\[Lambda]_F_"<>ToString[i]->"thread_F_"<>ToString[i],
 "\[Lambda]Process"->"\[Lambda]_B_"<>ToString[i]->"thread_B_"<>ToString[i],
 NetPort["contract_"<>ToString[i],"Shortcut"]->NetPort["expand_"<>ToString[i],"Shortcut"],
 NetPort["expand_"<>ToString[i],"Output"]->"thread_B_"<>ToString[i-1]->If[i>1,NetPort["expand_"<>ToString[i-1],"Input"],"postprocess"]},{i,depth}],
 "postprocess"->"Loss",
 NetPort["Noise"]->"Loss"->"Rescale",
 NetPort["Scale"]->"Rescale"->NetPort["Loss"]}],
 "NoiseP"->Prepend[size,channel],
 "\[Lambda]"->\[Lambda]latent]]
 
 (*train the unconditional diffusion model*)
 
validation=Block[{tempt=Import["/data01/users/xzn/BX/Diffusion/EastAsia_Validation_Data.mx"]},
 Table[Block[{zz=tempt[[year,region]],condition},
   Table[<|"NoiseP"->zz[[i,"NoiseP"]],
           "Noise"->zz[[i,"Noise"]],
           (*"P"->(zz[[i,"NoiseP"]]-zz[[i,"Noise]]*Sqrt[zz[[i,"scale"]]])/Sqrt[1-zz[[i,"scale"]]],*)
           "\[Lambda]"->zz[[i,"\[Lambda]"]],
           "Scale"->zz[[i,"scale"]]|>,{i,Length[zz]}]],{year,Length[tempt]},{region,Dimensions[tempt][[2]]}]];

GlobeLoss=Infinity;
Report[net_,loss_]:=Block[{},
        Print[{GlobeLoss,loss}];
        If[loss<GlobeLoss,Block[{},Print["Update"];
        Set[GlobeLoss,loss];
        Export["/data01/users/xzn/BX/Diffusion/DDPM_UnCondition_trained.mx",net]]]];


trained=NetTrain[UUNet,
 {Function[Block[{u,\[Lambda]s,noise,select,precipitation,elevation,dynamics,position,\[Lambda]embedding,nprecipitation,condition,scale},
  u=RandomReal[UniformDistribution[{0,1}],batch];
  \[Lambda]s=\[Lambda][u];
  scale=(\[Sigma][\[Lambda][u]])^2;
  noise=RandomReal[NormalDistribution[0,1],Join[{batch,channel},size]];
  select=Table[Block[{dPosition,dselect,pselect,pPosition},
        dPosition=Block[{tempt={RandomChoice[Range[1,Length[dlat]-dRange]],RandomChoice[Range[1,Length[dlon]-dRange]]}},
           {{tempt[[1]],tempt[[1]]+dRange-1},{tempt[[2]],tempt[[2]]+dRange-1}}];
        pPosition=(dPosition-1)*5+{{1,0},{1,0}};
       {Block[{year=RandomInteger[{1,Length[p]-test}]},{year,RandomInteger[{2,Length[p[[year]]]-1}]}],dPosition,pPosition}],batch];
  precipitation=Table[NumericArray[{p[[select[[i,1,1]]]][[select[[i,1,2]],select[[i,3,1,1]];;select[[i,3,1,2]],select[[i,3,2,1]];;select[[i,3,2,2]]]]},"Real16"],{i,batch}];
  \[Lambda]embedding=Map[\[Lambda]Encoding,\[Lambda]s];
  nprecipitation=Table[Fzx[\[Lambda]s[[i]],Normal[precipitation[[i]]],noise[[i]]],{i,batch}];
  condition=DownscalingNet[<|"Elevation"->elevation,"Dynamics"->dynamics|>,TargetDevice->"GPU"];
 <|"NoiseP"->nprecipitation,
   "Noise"->noise,
   "\[Lambda]"->\[Lambda]embedding,
   "Scale"->scale|>]],"RoundLength" ->batch*1000},
 LossFunction ->{"Loss"->Scaled[Prepend[size ,channel]/.List->Times]},
 BatchSize -> batch,
 MaxTrainingRounds->10000,
 ValidationSet->Flatten[validation],
 TargetDevice->{"GPU",All},
 Method->{"ADAM","LearningRate"->.01*10^-3,"L2Regularization"->10^-6},
 TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];
 
 
 (*conditional diffusion model*)
 UNet=NetInitialize[NetGraph[<|Table[{"contract_"<>ToString[i]->contract[c*2^(i-1)],
                      "expand_"<>ToString[i]->expand[c*2^Max[(i-1),1],size/2^(i-1)],
                      "\[Lambda]_F_"<>ToString[i]->{LinearLayer[c*2^(i-1)],ReplicateLayer[Floor[size/2^i],2]},
                      "\[Lambda]_B_"<>ToString[i]->{LinearLayer[c*2^(i-1)],ReplicateLayer[Floor[size/2^i],2]},
                      "thread_F_"<>ToString[i]->ThreadingLayer[Plus],
                      "thread_B_"<>ToString[i]->ThreadingLayer[Plus]},{i,depth}],
                "\[Lambda]_F_0"->{LinearLayer[c],ReplicateLayer[size,2]},"thread_F_0"->ThreadingLayer[Plus],
                "\[Lambda]_B_0"->{LinearLayer[c],ReplicateLayer[size,2]},"thread_B_0"->ThreadingLayer[Plus],
                "preprocess"->ConvolutionLayer[c,{1,1}],
                "postprocess"->{ConvolutionLayer[c,{1,1}],NormalizationLayer[],ElementwiseLayer["GELU"],ConvolutionLayer[channel,{1,1}]},
                "ubase"->res[c*2^(depth-1)],
                "cateI"->CatenateLayer[],
                "\[Lambda]Process"->{LinearLayer[2\[Lambda]latent],ElementwiseLayer["GELU"],LinearLayer[3\[Lambda]latent],ElementwiseLayer["GELU"]},
                "Loss"->MeanSquaredLossLayer[],
                "Rescale"->ThreadingLayer[Times]|>,
 Flatten[{
 NetPort["Condition"]->"cateI",
 NetPort["NoiseP"]->"cateI"->"preprocess"->"thread_F_0"->"contract_1","preprocess"->"thread_B_0",
 NetPort["\[Lambda]"]->"\[Lambda]Process"->"\[Lambda]_F_0"->"thread_F_0","\[Lambda]Process"->"\[Lambda]_B_0"->"thread_B_0",
 Table[{NetPort["contract_"<>ToString[i],"Pooling"]->"thread_F_"<>ToString[i]->
 If[i<depth,"contract_"<>ToString[i+1],"ubase"->"thread_B_"<>ToString[depth]->NetPort["expand_"<>ToString[depth],"Input"]],
 "\[Lambda]Process"->"\[Lambda]_F_"<>ToString[i]->"thread_F_"<>ToString[i],
 "\[Lambda]Process"->"\[Lambda]_B_"<>ToString[i]->"thread_B_"<>ToString[i],
 NetPort["contract_"<>ToString[i],"Shortcut"]->NetPort["expand_"<>ToString[i],"Shortcut"],
 NetPort["expand_"<>ToString[i],"Output"]->"thread_B_"<>ToString[i-1]->If[i>1,NetPort["expand_"<>ToString[i-1],"Input"],"postprocess"]},{i,depth}],
 "postprocess"->"Loss",
 NetPort["Noise"]->"Loss"->"Rescale",
 NetPort["Scale"]->"Rescale"->NetPort["Loss"]}],
 "NoiseP"->Prepend[size,channel],
 "Condition"->Prepend[size,channel],
 "\[Lambda]"->\[Lambda]latent]]
 
 (*train the conditional diffusion model*)
 GlobeLoss=Infinity;
Report[net_,loss_]:=Block[{},
        Print[{GlobeLoss,loss}];
        If[loss<GlobeLoss,Block[{},Print["Update"];
        Set[GlobeLoss,loss];
        Export["/data01/users/xzn/BX/Diffusion/DDPM_trained.mx",net]]]];

trained=NetTrain[UNet,
 {Function[Block[{u,\[Lambda]s,noise,select,precipitation,elevation,dynamics,position,\[Lambda]embedding,nprecipitation,condition,scale},
  u=RandomReal[UniformDistribution[{0,1}],batch];
  \[Lambda]s=\[Lambda][u];
  scale=(\[Sigma][\[Lambda][u]])^2;
  noise=RandomReal[NormalDistribution[0,1],Join[{batch,channel},size]];
  select=Table[Block[{dPosition,dselect,pselect,pPosition},
        dPosition=Block[{tempt={RandomChoice[Range[1,Length[dlat]-dRange]],RandomChoice[Range[1,Length[dlon]-dRange]]}},
           {{tempt[[1]],tempt[[1]]+dRange-1},{tempt[[2]],tempt[[2]]+dRange-1}}];
        pPosition=(dPosition-1)*5+{{1,0},{1,0}};
       {Block[{year=RandomInteger[{1,Length[p]-test}]},{year,RandomInteger[{2,Length[p[[year]]]-1}]}],dPosition,pPosition}],batch];
  precipitation=Table[NumericArray[{p[[select[[i,1,1]]]][[select[[i,1,2]],select[[i,3,1,1]];;select[[i,3,1,2]],select[[i,3,2,1]];;select[[i,3,2,2]]]]},"Real16"],{i,batch}];
  elevation=Table[ele[[;;,select[[i,3,1,1]];;select[[i,3,1,2]],select[[i,3,2,1]];;select[[i,3,2,2]]]],{i,batch}];
  dynamics=Table[Transpose[d[[select[[i,1,1]]]][[(select[[i,1,2]]-2)*3+1;;(select[[i,1,2]]+1)*3,;;,select[[i,2,1,1]];;select[[i,2,1,2]],select[[i,2,2,1]];;select[[i,2,2,2]]]]],{i,batch}];
  \[Lambda]embedding=Map[\[Lambda]Encoding,\[Lambda]s];
  nprecipitation=Table[Fzx[\[Lambda]s[[i]],Normal[precipitation[[i]]],noise[[i]]],{i,batch}];
  condition=DownscalingNet[<|"Elevation"->elevation,"Dynamics"->dynamics|>,TargetDevice->"GPU"];
 <|"NoiseP"->nprecipitation,
   "Noise"->noise,
   "\[Lambda]"->\[Lambda]embedding,
   "Condition"->condition,
   "Scale"->scale|>]],"RoundLength" ->batch*1000},
 LossFunction ->{"Loss"->Scaled[Prepend[size ,channel]/.List->Times]},
 BatchSize -> batch,
 MaxTrainingRounds->10000,
 ValidationSet->Flatten[validation],
 TargetDevice->{"GPU",All},
 Method->{"ADAM","LearningRate"->.01*10^-3,"L2Regularization"->10^-6},
 TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];
 
 
 
 (*sample generation*)
net=NetTake[Import["/data01/users/xzn/BX/Diffusion/DDPM_trained.mx"],"postprocess"];
unet=NetTake[Import["/data01/users/xzn/BX/Diffusion/DDPM_UnCondition_trained.mx"],"postprocess"];
DownscalingNet=NetTake[Import["/data01/users/xzn/BX/Diffusion/Downscaling_Net.mx"],"postprocess_0"];
data=Import["/data01/users/xzn/BX/Diffusion/EastAsia_Validation_Data.mx"];

year=3;
region=10;
id=694;

case=data[[year,region,id]];
dynamics=Normal[case["Dynamics"]];
ele=Normal[case["Elevation"]];
p=(case[["NoiseP"]]-case[["Noise"]]*Sqrt[case[["scale"]]])/Sqrt[1-case[["scale"]]];
condition=DownscalingNet[<|"Dynamics"->dynamics,"Elevation"->ele|>];

ss=0.2;
Report[net_]:=Block[{batch=24,\[Lambda]seq,\[Sigma]Seq,initial,noisenet},
 \[Lambda]seq=Table[\[Lambda][i],{i,.9945,0.001,-1*10^-3}];
 \[Sigma]seq[v_]:=Sqrt[(1-Exp[\[Lambda]seq[[1;;-2]]-\[Lambda]seq[[2;;]]])](\[Sigma][\[Lambda]seq[[1;;-2]]])^(1-v)*\[Sigma][\[Lambda]seq[[2;;-1]]]^v;
 \[Alpha]seq=Map[\[Alpha],\[Lambda]seq];
 initial=RandomReal[NormalDistribution[0,1],Join[{batch,channel},size]];
 condition=DownscalingNet[<|"Dynamics"->dynamics,"Elevation"->ele|>];
 Table[Set[initial,
        (initial-((1+ss)*net[<|"NoiseP"->initial,
                    "\[Lambda]"->Table[\[Lambda]Encoding[\[Lambda]seq[[t]]],batch],
                    "Scale"->Table[(\[Sigma][\[Lambda]seq[[t]]])^2,batch],
                    "Condition"->Table[condition,batch]|>,TargetDevice->"GPU"]
                  -ss*unet[<|"NoiseP"->initial,
                    "\[Lambda]"->Table[\[Lambda]Encoding[\[Lambda]seq[[t]]],batch]|>,TargetDevice->"GPU"])*(1-\[Alpha]seq[[t]]^2/\[Alpha]seq[[t+1]]^2)/Sqrt[1-\[Alpha]seq[[t]]^2])/\[Alpha]seq[[t]]*\[Alpha]seq[[t+1]]+
       RandomReal[NormalDistribution[0,1],Join[{batch,channel},size]]*\[Sigma]seq[.5][[t]]];
       Print[{t,Map[MinMax,initial[[1;;-1;;5]]]}];
       initial,{t,Length[\[Lambda]seq]-1}][[-1]]];
       
result=Report[net];
Export["/data01/users/xzn/BX/Diffusion/tempt.mx",{result,p,condition}];


 
 



                      
