function saveEx()
    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');

    b{1000} = [];
    fileID = fopen('data.txt','w');
    for i=1:1000
        b{i} = {labels(i), images(:, i)};
        allOneString = sprintf('%.0f,' , images(:, i));
        allOneString = allOneString(1:end-1);% strip final comma
        fprintf(fileID, [num2str(labels(i)) ';']);
        fprintf(fileID, allOneString);
        fprintf(fileID, '\r\n');
        
    end
    fclose(fileID);

    
end