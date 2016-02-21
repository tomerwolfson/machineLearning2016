function save_map( filename, map_container )

keySet = keys(map_container);
values = zeros(size(keySet));
for i = 1:length(keySet)
    key = keySet{i};
    values(i) = map_container(key);
end
save(filename,'keySet','values');
end

