function [ output_args ] = drawAnnotations( img, allClasses, outputFile,annotTable )
% Draws annotated cells on image
figure
imshow(img)
hold on
for classV = allClasses'
    color='black';
    if(strcmp(classV,'c'))
        color='green';
    elseif(strcmp(classV,'o'))
        color='red';
    elseif(strcmp(classV,'l'))
        color='blue';
    elseif(strcmp(classV,'e'))
        color='yellow';
    end
    plot(annotTable.x(strcmp(annotTable.class,classV)),annotTable.y(strcmp(annotTable.class,classV)),'.','MarkerSize',10,'MarkerEdgeColor',color)
end
saveas(gcf,outputFile)


end

