clear all;
clc;
clf;

dy_datas = load("./output_gp_dy_0.txt");
scatter3(-dy_datas(:,2),dy_datas(:,3),dy_datas(:,4), 10, dy_datas(:,5:7)*4, 'filled');
axis equal;

% xlim([-0.2 0.4]);
ylim([650 680]);
% zlim([-0.4 0])


% Initialize empty array for filtered data
filtered_data = [];

% Get number of rows
num_rows = size(dy_datas, 1);

% For loop to extract rows where y (column 3) is between 650 and 680
for i = 1:num_rows
    y_value = dy_datas(i, 3);
    if y_value >= 650 && y_value <= 680
        filtered_data = [filtered_data; dy_datas(i, :)];
    end
end

% Save filtered data to text file (choose one method):

% Method 3: For MATLAB R2019a and later (recommended)
writematrix(filtered_data, 'filtered_output.txt', 'Delimiter', 'comma');
