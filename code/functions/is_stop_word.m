function result = is_stop_word(str)
    % returns whether str is an english stop word
    %
    % list of english stop words taken from:
    % http://www.textfixer.com/resources/common-english-words.txt
    % http://blogs.mathworks.com/loren/2014/06/04/analyzing-twitter-with-matlab/#1d9f294f-e18d-468e-b2fc-494f10def545
    %
    %full list:
    % a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your
    
    stop_string = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your';
    stop_words = strsplit(stop_string, ',');
    n = size(stop_words, 2)
    
    result = 0;
    for i = 1:n
        %check for each stop word if it equals str
        if (strcmp(str, stop_words(i)):
            result = 1;
            return
    end

end