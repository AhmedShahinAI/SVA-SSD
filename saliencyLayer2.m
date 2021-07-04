    classdef saliencyLayer2 < nnet.layer.Layer

        properties(Learnable)
            % (Optional) Layer properties.
         %   NumOutputs
          alpha
        end
       
       

   methods
        function layer = saliencyLayer2(numChannels,name)
            % Set number of inputs.
          %  layer.NumOutputs = numOutputs;
            % Set layer name.
            layer.Name = name;
          %  layer.NumChannels=NumChannels;
            
           % Set layer description.
            layer.Description = "grad with " + numChannels + " channels";
            % Set layer description.
            layer.Description = "Salieny Layer";
            
            
            
             minalpha=1;
             maxalpha=1.5;

        %     
           layer.alpha = abs(minalpha + (maxalpha-minalpha).*rand(1,1));
            
            
        end
    
        function Z = predict(layer,X)
               
            
         sz= size(X);
         chk=length(sz);
         
%          size(m)
          if chk==4
             
         patchsz=sz(4);
         
         patches=extractdata(X);  
%          size(patches)
%          class(patches)
         m=zeros(sz(1), sz(2), sz(3) ,sz(4));
         
    for i=1:patchsz
            
           img= ( patches(:,:,:,i));  
          % img=rgb2xyz(img);
           
            grayimg=rgb2gray(img);
            FFT = fft2(grayimg); 
            LogAmplitude = log(abs(FFT));
            Phase = angle(FFT);
            SpectralResidual = LogAmplitude - imfilter(LogAmplitude, fspecial('average', 3), 'replicate'); 
            saliencyMap = abs(ifft2(exp(SpectralResidual + i*Phase))).^2;
% %%After Effect
            saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('disk', 3)));
    
           z=imadjust(saliencyMap,[]);
        
           z=histeq(z);
            
           %bw=im2bw(grayimg,0.3);
           
%     se = strel('disk',5,0);
%     Ie = imerode(grayimg , se);
%     z = imreconstruct(Ie,grayimg);
           
            
            
            zz=cat(3,z,z,z);
                        
            m(:,:,:,i)=zz;


    end     
     
        t=dlarray(single(m));
        Z=layer.alpha.*t;
     %    Z=X;
       else

            
        Z=X;
        end
            
            
        end
% 
%         function [dLdX] = backward(~,~,~,~,~)
%         end
% 


   end
    
   
   
  
   
   
    end
    
    
    
    
    