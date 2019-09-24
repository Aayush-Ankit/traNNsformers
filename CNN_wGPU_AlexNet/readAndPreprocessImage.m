function Iout = readAndPreprocessImage(filename)


try
    I = imread(filename);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);
end

% Resize the image as required for the CNN.
Iout = imresize(I, [227 227]);

catch
    Iout = rand([227 227 3]);
    %fprintf(DEBUGfid, ['CMYK image detected.', '\n']);
end

end