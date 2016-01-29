function [new_str] = parse_review_string(str)
    %initial parsing of the review string
    %before the vectorization process

    %removes from the review string unnesscary symbols:
    % numbers, punctuation symbols, etc.
    new_str = str;
    new_str = regexprep(new_str, '_', ' '); %replace underscore with ' '
    new_str = regexprep(new_str, '[^\w\s]', ''); %replace non character or digit symbols with ''
    new_str = regexprep(new_str, '[\d]', ''); %replace digit symbols with ''
    
    %convert string characters to lowercase
    new_str = lower(new_str);
    
end