function helperDisplayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D',...
    'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',...
    'V','W','X','Y','Z','-'];
colHeadings = arrayfun(@(x)sprintf('%s',x),digits,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'digit  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end