count=$(ls -1 database/simlingo/*.tar.gz 2>/dev/null | wc -l)
i=0

for file in database/simlingo/*.tar.gz; do
    [ -f "$file" ] || continue
    i=$((i + 1))
    # echo "[$i/$count] Extracting $file to database/simlingo/..."
    if tar -xzf "$file" -C database/simlingo/; then
        # rm -f "$file"
        echo "[$i/$count] Done with $file."
    else
        # rm -f "$file"
        echo "[$i/$count] Failed to extract $file"
    fi
done
