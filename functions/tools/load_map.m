function [ map_container ] = load_map( filename )
load(filename);
map_container = containers.Map(keySet,values);
end

