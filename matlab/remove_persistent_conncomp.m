function processed = remove_persistent_conncomp(processed,persistent_objects)

L = labelmatrix(processed.im_cleaned_cc);

% Identify components that overlap with persistent objects
overlap = unique(L(persistent_objects));
to_remove = ismember(L, overlap);

processed.im_cleaned2 = processed.im_cleaned;
processed.im_cleaned2(to_remove) = false;
processed.im_cleaned2_cc = bwconncomp(processed.im_cleaned2);

end