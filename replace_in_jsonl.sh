#remove invalid references with garbage bibtex entities
grep '\\\\' $1  | wc -l
sed -i '/\\\\/d' $1
grep '\\\\' $1  | wc -l
