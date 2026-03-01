#!/bin/bash

# Default values
NAME="World"
VERBOSE=false

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -n|--name)
            NAME="$2"
            shift # Shift past the argument value
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # Shift past the argument name
done

# Use the parsed arguments
if $VERBOSE; then
    echo "Verbose mode enabled."
fi

echo "Hello, $NAME!"