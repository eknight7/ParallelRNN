load('data_batch_1.mat');

T = 10;
num_input = 1024;
num_hidden = 1024;
num_output = 1024;
batch_size = 1024;
X = zeros([T, batch_size, num_input]);
target = zeros([T, batch_size, num_output]);
for i=1:batch_size
    I = double(rgb2gray(reshape(data(i,:),[32,32,3])))/255;
    prevI = I;
    rot_angle = 10;
    for t=1:T
        newI = imrotate(I, t*rot_angle, 'crop');
        X(t,i,:) = reshape(prevI, [num_input,1]);
        target(t,i,:) = reshape(newI, [num_output, 1]);
        prevI = newI;
    end
end

for i=1:T
    imwrite(reshape(X(i,:,:),[batch_size, num_input]), sprintf('X_%d.png', i));
    imwrite(reshape(target(i,:,:),[batch_size, num_output]), sprintf('T_%d.png', i));
end

imwrite(zeros([batch_size, num_hidden]), 'h0.png');
imwrite(rand([num_input, num_hidden]), 'Wxh.png');
imwrite(rand([num_hidden, num_hidden]), 'Whh.png');
imwrite(rand([num_hidden, num_output]), 'Why.png');
