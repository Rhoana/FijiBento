package org.janelia.alignment.spec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.janelia.alignment.spec.validator.TileSpecValidator;
import org.janelia.alignment.util.ProcessTimer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A collection of tile specifications that also includes all referenced transform specifications,
 * allowing the tile specifications to be fully resolved.
 *
 * @author Eric Trautman
 */
public class ResolvedTileSpecCollection {

    private final Map<String, TransformSpec> transformIdToSpecMap;
    private final Map<String, TileSpec> tileIdToSpecMap;

    private transient TileSpecValidator tileSpecValidator;

    @SuppressWarnings("UnusedDeclaration")
    public ResolvedTileSpecCollection() {
        this(new ArrayList<TransformSpec>(), new ArrayList<TileSpec>());
    }

    /**
     * Creates a collection containing the provided transform and tile specs.
     *
     * @param  transformSpecs  shared (referenced) transform specifications.
     * @param  tileSpecs       tile specifications.
     *
     * @throws IllegalArgumentException
     *   if any of the tile specifications reference an unknown transform specification.
     */
    public ResolvedTileSpecCollection(final Collection<TransformSpec> transformSpecs,
                                      final Collection<TileSpec> tileSpecs)
            throws IllegalArgumentException {

        this.transformIdToSpecMap = new HashMap<>(transformSpecs.size() * 2);
        this.tileIdToSpecMap = new HashMap<>(tileSpecs.size() * 2);
        this.tileSpecValidator = null;

        for (final TransformSpec transformSpec : transformSpecs) {
            addTransformSpecToCollection(transformSpec);
        }

        for (final TileSpec tileSpec : tileSpecs) {
            addTileSpecToCollection(tileSpec);
        }
    }

    /**
     * Sets the tile spec validator for this collection (a null value disables validation).
     * Any tile spec identified as invalid by this validator will be removed / filtered from the collection.
     *
     * @param  tileSpecValidator  validator to use for all tile specs.
     */
    public void setTileSpecValidator(final TileSpecValidator tileSpecValidator) {
        this.tileSpecValidator = tileSpecValidator;
    }

    /**
     * @param  tileId  identifier for the desired tile.
     *
     * @return the tile specification with the specified id (or null if it does not exist).
     */
    public TileSpec getTileSpec(final String tileId) {
        return tileIdToSpecMap.get(tileId);
    }

    /**
     * @return the set of resolved tile specifications in this collection.
     *
     * @throws IllegalArgumentException
     *   if any of the tile specifications reference an unknown transform specification.
     */
    public Collection<TileSpec> getTileSpecs()
            throws IllegalArgumentException {
        resolveTileSpecs(); // this needs to be done here for collections deserialized from JSON
        return tileIdToSpecMap.values();
    }

    /**
     * Adds a tile specification to this collection and verifies that
     * any referenced transforms already exist in this collection.
     *
     * @param  tileSpec  spec to add.
     *
     * @throws IllegalArgumentException
     *   if the tile specification references an unknown transform specification.
     */
    public void addTileSpecToCollection(final TileSpec tileSpec)
            throws IllegalArgumentException {
        resolveTileSpec(tileSpec);
        tileIdToSpecMap.put(tileSpec.getTileId(), tileSpec);
    }

    /**
     * @return the set of shared transform specifications in this collection.
     */
    public Collection<TransformSpec> getTransformSpecs() {
        return transformIdToSpecMap.values();
    }

    /**
     * Adds a transform specification to this collection's set of shared transformations.
     *
     * @param  transformSpec  spec to add.
     */
    public void addTransformSpecToCollection(final TransformSpec transformSpec) {
        transformIdToSpecMap.put(transformSpec.getId(), transformSpec);
    }

    /**
     * Adds a transform specification to the specified tile.
     *
     * The tile's bounding box is recalculated after the new transform is applied.
     *
     * If this collection has a tile spec validator that determines the spec is invalid
     * (after applying the transform), the spec will be removed from the collection.
     *
     * @param  tileId         identifies the tile to which the transform should be added.
     *
     * @param  transformSpec  the transform to add.
     *
     * @param  replaceLast    if true, the specified transform will replace the tile's last transform;
     *                        otherwise, the specified transform will simply be appended.
     *
     * @throws IllegalArgumentException
     *   if the specified tile cannot be found or the specified transform cannot be fully resolved.
     */
    public void addTransformSpecToTile(final String tileId,
                                       final TransformSpec transformSpec,
                                       final boolean replaceLast) throws IllegalArgumentException {

        final TileSpec tileSpec = tileIdToSpecMap.get(tileId);

        if (tileSpec == null) {
            throw new IllegalArgumentException("tile spec with id '" + tileId + "' not found");
        }

        if (! transformSpec.isFullyResolved()) {
            transformSpec.resolveReferences(transformIdToSpecMap);
            if (! transformSpec.isFullyResolved()) {
                throw new IllegalArgumentException("transform spec references the following unresolved transform ids " +
                                                   transformSpec.getUnresolvedIds());
            }
        }

        if (replaceLast) {
            tileSpec.removeLastTransformSpec();
        }

        tileSpec.addTransformSpecs(Arrays.asList(transformSpec));

        // addition of new transform spec obsolesces the previously resolved coordinate transform instance,
        // so we need to re-resolve the tile before re-deriving the bounding box
        resolveTileSpec(tileSpec);

        tileSpec.deriveBoundingBox(tileSpec.getMeshCellSize(), true);

        if (tileSpecValidator != null) {
            removeTileIfInvalid(tileSpec);
        }
    }

    /**
     * Adds a reference to the specified transform to all tiles in this collection.
     *
     * Each tile's bounding box is recalculated after the new transform is applied
     * (so this can potentially be a long running operation).
     *
     * If this collection has a tile spec validator that determines one or more tile specs are invalid
     * (after applying the transform), those tile specs will be removed from the collection.
     *
     * @param  transformId         identifies the transform to be applied to all tiles.
     *
     * @throws IllegalArgumentException
     *   if the specified transform cannot be found.
     */
    public void addReferenceTransformToAllTiles(final String transformId)
            throws IllegalArgumentException {

        final TransformSpec transformSpec = transformIdToSpecMap.get(transformId);
        if (transformSpec == null) {
            throw new IllegalArgumentException("transform " + transformId + " cannot be found");
        }

        final TransformSpec referenceTransformSpec = new ReferenceTransformSpec(transformId);

        final ProcessTimer timer = new ProcessTimer();
        int tileSpecCount = 0;
        for (final String tileId : tileIdToSpecMap.keySet()) {
            addTransformSpecToTile(tileId, referenceTransformSpec, false);
            tileSpecCount++;
            if (timer.hasIntervalPassed()) {
                LOG.info("addReferenceTransformToAllTiles: added transform to {} out of {} tiles",
                         tileSpecCount, tileIdToSpecMap.size());
            }
        }

        LOG.info("addReferenceTransformToAllTiles: added transform to {} tiles, elapsedSeconds={}",
                 tileSpecCount, timer.getElapsedSeconds());
    }

    /**
     * Verifies that all tile specs in this collection have the specified z value.
     *
     * @param  expectedZ  the expected z value for all tiles.
     *
     * @throws IllegalArgumentException
     *   if the z value for any tile is null or does not match the expected z value.
     */
    public void verifyAllTileSpecsHaveZValue(final double expectedZ)
            throws IllegalArgumentException {
        Double actualZ;
        for (final TileSpec tileSpec : tileIdToSpecMap.values()) {
            actualZ = tileSpec.getZ();
            if (actualZ == null) {
                throw new IllegalArgumentException(getBadTileZValueMessage(expectedZ, tileSpec));
            } else {
                if (Double.compare(expectedZ, actualZ) != 0) {
                    throw new IllegalArgumentException(getBadTileZValueMessage(expectedZ, tileSpec));
                }
            }
        }
    }

    /**
     * Removes any tile specs not identified in the provided set from this collection.
     *
     * @param  tileIdsToKeep  identifies which tile specs should be kept.
     */
    public void filterSpecs(final Set<String> tileIdsToKeep) {
        final Iterator<Map.Entry<String, TileSpec>> i = tileIdToSpecMap.entrySet().iterator();
        Map.Entry<String, TileSpec> entry;
        while (i.hasNext()) {
            entry = i.next();
            if (! tileIdsToKeep.contains(entry.getKey())) {
                i.remove();
            }
        }

        // TODO: remove any unreferenced transforms
    }

    /**
     * @return the number opf transform specs in this collection.
     */
    public int getTransformCount() {
        return transformIdToSpecMap.size();
    }

    /**
     * @return the number opf tile specs in this collection.
     */
    public int getTileCount() {
        return tileIdToSpecMap.size();
    }

    /**
     * @return true if this collection has at least one tile spec; otherwise false.
     */
    public boolean hasTileSpecs() {
        return tileIdToSpecMap.size() > 0;
    }

    /**
     * Resolves referenced transform specs for all tile specs in this collection.
     *
     * @throws IllegalArgumentException
     *   if a transform spec reference cannot be resolved.
     */
    public void resolveTileSpecs()
            throws IllegalArgumentException {
        for (final TileSpec tileSpec : tileIdToSpecMap.values()) {
            resolveTileSpec(tileSpec);
        }
    }

    @Override
    public String toString() {
        return "{transformCount: " + getTransformCount() +
               ", tileCount: " + getTileCount() +
               '}';
    }

    private String getBadTileZValueMessage(final double expectedZ,
                                           final TileSpec tileSpec) {
        return "all tiles must have a z value of " + expectedZ + " but tile " +
               tileSpec.getTileId() + " has a z value of " + tileSpec.getZ();
    }

    private void resolveTileSpec(final TileSpec tileSpec)
            throws IllegalArgumentException {
        final ListTransformSpec transforms = tileSpec.getTransforms();
        if (! transforms.isFullyResolved()) {
            transforms.resolveReferences(transformIdToSpecMap);
            if (! transforms.isFullyResolved()) {
                throw new IllegalArgumentException("tile " + tileSpec.getTileId() +
                                                   " requires the following transform ids " +
                                                   transforms.getUnresolvedIds());
            }
        }
    }

    private void removeTileIfInvalid(final TileSpec tileSpec) {

        try {
            tileSpecValidator.validate(tileSpec);
        } catch (final IllegalArgumentException e) {
            LOG.error(e.getMessage());
            tileIdToSpecMap.remove(tileSpec.getTileId());
        }

    }

    private static final Logger LOG = LoggerFactory.getLogger(ResolvedTileSpecCollection.class);
}