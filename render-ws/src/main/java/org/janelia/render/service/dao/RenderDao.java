package org.janelia.render.service.dao;

import com.mongodb.BasicDBList;
import com.mongodb.MongoClient;
import com.mongodb.QueryOperators;
import com.mongodb.bulk.BulkWriteResult;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.BulkWriteOptions;
import com.mongodb.client.model.IndexOptions;
import com.mongodb.client.model.InsertOneModel;
import com.mongodb.client.model.ReplaceOneModel;
import com.mongodb.client.model.UpdateOptions;
import com.mongodb.client.model.WriteModel;
import com.mongodb.client.result.DeleteResult;
import com.mongodb.client.result.UpdateResult;

import java.io.IOException;
import java.io.OutputStream;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import javax.annotation.Nonnull;

import org.bson.Document;
import org.janelia.alignment.RenderParameters;
import org.janelia.alignment.spec.Bounds;
import org.janelia.alignment.spec.ListTransformSpec;
import org.janelia.alignment.spec.ResolvedTileSpecCollection;
import org.janelia.alignment.spec.SectionData;
import org.janelia.alignment.spec.TileBounds;
import org.janelia.alignment.spec.TileCoordinates;
import org.janelia.alignment.spec.TileSpec;
import org.janelia.alignment.spec.TransformSpec;
import org.janelia.alignment.spec.stack.StackId;
import org.janelia.alignment.spec.stack.StackMetaData;
import org.janelia.alignment.spec.stack.StackStats;
import org.janelia.alignment.util.ProcessTimer;
import org.janelia.render.service.model.ObjectNotFoundException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Data access object for Render database.
 *
 * @author Eric Trautman
 */
public class RenderDao {

    public static final String RENDER_DB_NAME = "render";
    public static final String STACK_META_DATA_COLLECTION_NAME = "admin__stack_meta_data";

    public static RenderDao build()
            throws UnknownHostException {
        final MongoClient mongoClient = SharedMongoClient.getInstance();
        return new RenderDao(mongoClient);
    }

    private final MongoDatabase renderDatabase;

    public RenderDao(final MongoClient client) {
        renderDatabase = client.getDatabase(RENDER_DB_NAME);
    }

    /**
     * @return a render parameters object for all tiles that match the specified criteria.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public RenderParameters getParameters(final StackId stackId,
                                          final String groupId,
                                          final Double x,
                                          final Double y,
                                          final Double z,
                                          final Integer width,
                                          final Integer height,
                                          final Double scale)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("x", x);
        validateRequiredParameter("y", y);
        validateRequiredParameter("z", z);
        validateRequiredParameter("width", width);
        validateRequiredParameter("height", height);
        validateRequiredParameter("scale", scale);

        final double lowerRightX = x + width;
        final double lowerRightY = y + height;
        final Document tileQuery = getIntersectsBoxQuery(z, x, y, lowerRightX, lowerRightY);
        if (groupId != null) {
            tileQuery.append("groupId", groupId);
        }

        final RenderParameters renderParameters = new RenderParameters(null, x, y, width, height, scale);
        addResolvedTileSpecs(stackId, tileQuery, renderParameters);

        return renderParameters;
    }

    /**
     * @return a render parameters object for all tiles that match the specified criteria.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public RenderParameters getParameters(final StackId stackId,
                                          final Double z,
                                          Double scale)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        if (scale == null) {
            scale = 1.0;
        }

        final Bounds bounds = getLayerBounds(stackId, z);
        final Double x = bounds.getMinX();
        final Double y = bounds.getMinY();
        final Double width = bounds.getMaxX() - x;
        final Double height = bounds.getMaxY() - y;

        final Document tileQuery = new Document("z", z);

        final RenderParameters renderParameters =
                new RenderParameters(null, x, y, width.intValue(), height.intValue(), scale);
        addResolvedTileSpecs(stackId, tileQuery, renderParameters);

        return renderParameters;
    }

    /**
     * @return number of tiles within the specified bounding box.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public long getTileCount(final StackId stackId,
                             final Double x,
                             final Double y,
                             final Double z,
                             final Integer width,
                             final Integer height)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("x", x);
        validateRequiredParameter("y", y);
        validateRequiredParameter("z", z);
        validateRequiredParameter("width", width);
        validateRequiredParameter("height", height);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        final double lowerRightX = x + width;
        final double lowerRightY = y + height;
        final Document tileQuery = getIntersectsBoxQuery(z, x, y, lowerRightX, lowerRightY);

        final long count = tileCollection.count(tileQuery);

        LOG.debug("getTileCount: found {} tile spec(s) for {}.find({})",
                  count, getFullName(tileCollection), tileQuery.toJson());

        return count;
    }

    /**
     * @return the specified tile spec.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing.
     *
     * @throws ObjectNotFoundException
     *   if a spec with the specified z and tileId cannot be found.
     */
    public TileSpec getTileSpec(final StackId stackId,
                                final String tileId,
                                final boolean resolveTransformReferences)
            throws IllegalArgumentException,
                   ObjectNotFoundException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("tileId", tileId);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        final Document query = new Document();
        query.put("tileId", tileId);

        LOG.debug("getTileSpec: {}.find({})", getFullName(tileCollection), query.toJson());

        // EXAMPLE:   find({ "tileId" : "140723171842050101.3299.0"})
        // INDEX:     tileId_1
        final Document document = tileCollection.find(query).first();

        if (document == null) {
            throw new ObjectNotFoundException("tile spec with id '" + tileId + "' does not exist in the " +
                                              getFullName(tileCollection) + " collection");
        }

        final TileSpec tileSpec = TileSpec.fromJson(document.toJson());

        if (resolveTransformReferences) {
            resolveTransformReferencesForTiles(stackId, Collections.singletonList(tileSpec));
        }

        return tileSpec;
    }

    public Map<String, TransformSpec> resolveTransformReferencesForTiles(final StackId stackId,
                                                                         final List<TileSpec> tileSpecs)
            throws IllegalStateException {

        final Set<String> unresolvedIds = new HashSet<>();
        ListTransformSpec transforms;
        for (final TileSpec tileSpec : tileSpecs) {
            transforms = tileSpec.getTransforms();
            if (transforms != null) {
                transforms.addUnresolvedIds(unresolvedIds);
            }
        }

        final Map<String, TransformSpec> resolvedIdToSpecMap = new HashMap<>();

        final int unresolvedCount = unresolvedIds.size();
        if (unresolvedCount > 0) {

            final MongoCollection<Document> transformCollection = getTransformCollection(stackId);
            getDataForTransformSpecReferences(transformCollection, unresolvedIds, resolvedIdToSpecMap, 1);

            // resolve any references within the retrieved transform specs
            for (final TransformSpec transformSpec : resolvedIdToSpecMap.values()) {
                transformSpec.resolveReferences(resolvedIdToSpecMap);
            }

            // apply fully resolved transform specs to tiles
            for (final TileSpec tileSpec : tileSpecs) {
                transforms = tileSpec.getTransforms();
                transforms.resolveReferences(resolvedIdToSpecMap);
                if (! transforms.isFullyResolved()) {
                    throw new IllegalStateException("tile spec " + tileSpec.getTileId() +
                                                    " is not fully resolved after applying the following transform specs: " +
                                                    resolvedIdToSpecMap.keySet());
                }
            }

        }

        return resolvedIdToSpecMap;
    }


    /**
     * @return a list of resolved tile specifications for all tiles that encompass the specified coordinates.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing, if the stack cannot be found, or
     *   if no tile can be found that encompasses the coordinates.
     */
    public List<TileSpec> getTileSpecs(final StackId stackId,
                                       final Double x,
                                       final Double y,
                                       final Double z)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("x", x);
        validateRequiredParameter("y", y);
        validateRequiredParameter("z", z);

        final Document tileQuery = getIntersectsBoxQuery(z, x, y, x, y);
        final RenderParameters renderParameters = new RenderParameters();
        addResolvedTileSpecs(stackId, tileQuery, renderParameters);

        if (! renderParameters.hasTileSpecs()) {
            throw new IllegalArgumentException("no tile specifications found in " + stackId +
                                               " for world coordinates x=" + x + ", y=" + y + ", z=" + z);
        }

        return renderParameters.getTileSpecs();
    }

    public void writeCoordinatesWithTileIds(final StackId stackId,
                                            final Double z,
                                            final List<TileCoordinates> worldCoordinatesList,
                                            final OutputStream outputStream)
            throws IllegalArgumentException, IOException {

        LOG.debug("writeCoordinatesWithTileIds: entry, stackId={}, z={}, worldCoordinatesList.size()={}",
                  stackId, z, worldCoordinatesList.size());

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final Document tileKeys = new Document("tileId", 1).append("_id", 0);

        // order tile specs by tileId to ensure consistent coordinate mapping
        final Document orderBy = new Document("tileId", 1);

        final ProcessTimer timer = new ProcessTimer();
        final byte[] openBracket = "[".getBytes();
        final byte[] comma = ",".getBytes();
        final byte[] closeBracket = "]".getBytes();

        int coordinateCount = 0;

        double[] world;
        Document tileQuery = new Document();
        MongoCursor<Document> cursor = null;
        Document document;
        Object tileId;
        String coordinatesJson;
        try {

            outputStream.write(openBracket);

            TileCoordinates worldCoordinates;
            for (int i = 0; i < worldCoordinatesList.size(); i++) {

                worldCoordinates = worldCoordinatesList.get(i);
                world = worldCoordinates.getWorld();

                if (world == null) {
                    throw new IllegalArgumentException("world values are missing for element " + i);
                } else if (world.length < 2) {
                    throw new IllegalArgumentException("world values must include both x and y for element " + i);
                }

                tileQuery = getIntersectsBoxQuery(z, world[0], world[1], world[0], world[1]);

                // EXAMPLE:   find({"z": 3299.0 , "minX": {"$lte": 95000.0}, "minY": {"$lte": 200000.0}, "maxX": {"$gte": 95000.0}, "maxY": {"$gte": 200000.0}}, {"tileId":1, "_id": 0}).sort({"tileId" : 1})
                // INDEXES:   z_1_minY_1_minX_1_maxY_1_maxX_1_tileId_1 (z1_minX_1, z1_maxX_1, ... used for edge cases)
                cursor = tileCollection.find(tileQuery).projection(tileKeys).sort(orderBy).iterator();

                if (i > 0) {
                    outputStream.write(comma);
                }
                outputStream.write(openBracket);

                if (cursor.hasNext()) {

                    document = cursor.next();
                    tileId = document.get("tileId");
                    if (tileId != null) {
                        worldCoordinates.setTileId(tileId.toString());
                    }
                    coordinatesJson = worldCoordinates.toJson();
                    outputStream.write(coordinatesJson.getBytes());

                    while (cursor.hasNext()) {
                        document = cursor.next();
                        tileId = document.get("tileId");
                        if (tileId != null) {
                            worldCoordinates.setTileId(tileId.toString());
                        }
                        coordinatesJson = worldCoordinates.toJson();

                        outputStream.write(comma);
                        outputStream.write(coordinatesJson.getBytes());
                    }

                } else {

                    coordinatesJson = worldCoordinates.toJson();
                    outputStream.write(coordinatesJson.getBytes());

                }

                cursor.close();

                outputStream.write(closeBracket);

                coordinateCount++;

                if (timer.hasIntervalPassed()) {
                    LOG.debug("writeCoordinatesWithTileIds: data written for {} coordinates", coordinateCount);
                }
            }

            outputStream.write(closeBracket);

        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }

        LOG.debug("writeCoordinatesWithTileIds: wrote data for {} coordinates returned by queries like {}.find({},{}).sort({}), elapsedSeconds={}",
                  coordinateCount, getFullName(tileCollection),
                  tileQuery.toJson(), tileKeys.toJson(), orderBy.toJson(), timer.getElapsedSeconds());
    }

    /**
     * @return a list of resolved tile specifications for all tiles that have the specified z.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or if the stack cannot be found, or
     *   if no tile can be found for the specified z.
     */
    public List<TileSpec> getTileSpecs(final StackId stackId,
                                       final Double z)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        final Document tileQuery = new Document("z", z);
        final RenderParameters renderParameters = new RenderParameters();
        addResolvedTileSpecs(stackId, tileQuery, renderParameters);

        if (! renderParameters.hasTileSpecs()) {
            throw new IllegalArgumentException("no tile specifications found in " + stackId +" for z=" + z);
        }

        return renderParameters.getTileSpecs();
    }

    /**
     * @return a resolved tile spec collection for all tiles that have the specified z.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or if the stack cannot be found, or
     *   if no tile can be found for the specified z.
     */
    public ResolvedTileSpecCollection getResolvedTiles(final StackId stackId,
                                                       final Double z)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        final Document tileQuery = new Document("z", z);
        final RenderParameters renderParameters = new RenderParameters();
        final Map<String, TransformSpec> resolvedIdToSpecMap = addResolvedTileSpecs(stackId,
                                                                                    tileQuery,
                                                                                    renderParameters);

        if (! renderParameters.hasTileSpecs()) {
            throw new IllegalArgumentException("no tile specifications found in " + stackId +" for z=" + z);
        }

        return new ResolvedTileSpecCollection(resolvedIdToSpecMap.values(),
                                              renderParameters.getTileSpecs());
    }

    /**
     * @return a resolved tile spec collection for all tiles that match the specified criteria.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or if the stack cannot be found, or
     *   if no tile can be found for the specified criteria.
     */
    public ResolvedTileSpecCollection getResolvedTiles(final StackId stackId,
                                                       final Double minZ,
                                                       final Double maxZ,
                                                       final String groupId,
                                                       final Double minX,
                                                       final Double maxX,
                                                       final Double minY,
                                                       final Double maxY)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);

        final Document query = getGroupQuery(minZ, maxZ, groupId, minX, maxX, minY, maxY);
        final RenderParameters renderParameters = new RenderParameters();
        final Map<String, TransformSpec> resolvedIdToSpecMap = addResolvedTileSpecs(stackId,
                                                                                    query,
                                                                                    renderParameters);

        if (! renderParameters.hasTileSpecs()) {
            throw new IllegalArgumentException("no tile specifications found in " + stackId +" for " + query);
        }

        return new ResolvedTileSpecCollection(resolvedIdToSpecMap.values(),
                                              renderParameters.getTileSpecs());
    }

    /**
     * Saves the specified tile spec to the database.
     *
     * @param  stackId            stack identifier.
     * @param  resolvedTileSpecs  collection of resolved tile specs (with referenced transforms).
     *
     * @throws IllegalArgumentException
     *   if any required parameters or transform spec references are missing.
     */
    public void saveResolvedTiles(final StackId stackId,
                                  final ResolvedTileSpecCollection resolvedTileSpecs)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("resolvedTileSpecs", resolvedTileSpecs);

        final Collection<TransformSpec> transformSpecs = resolvedTileSpecs.getTransformSpecs();
        final Collection<TileSpec> tileSpecs = resolvedTileSpecs.getTileSpecs();

        if (transformSpecs.size() > 0) {

            final MongoCollection<Document> transformCollection = getTransformCollection(stackId);

            final List<WriteModel<Document>> modelList = new ArrayList<>(transformSpecs.size());
            Document query = new Document();
            Document transformSpecObject;
            for (final TransformSpec transformSpec : transformSpecs) {
                query = new Document("id", transformSpec.getId());
                transformSpecObject = Document.parse(transformSpec.toJson());
                modelList.add(new ReplaceOneModel<>(query, transformSpecObject, UPSERT_OPTION));
            }

            final BulkWriteResult result = transformCollection.bulkWrite(modelList, UNORDERED_OPTION);

            if (LOG.isDebugEnabled()) {
                final String bulkResultMessage = getBulkResultMessage("transform specs", result, transformSpecs.size());
                LOG.debug("saveResolvedTiles: {} using {}.initializeUnorderedBulkOp()",
                          bulkResultMessage, getFullName(transformCollection), query.toJson());
            }

            // TODO: re-derive bounding boxes for all tiles (outside this collection) that reference modified transforms
        }

        if (tileSpecs.size() > 0) {

            final MongoCollection<Document> tileCollection = getTileCollection(stackId);

            final List<WriteModel<Document>> modelList = new ArrayList<>(tileSpecs.size());
            Document query = new Document();
            Document tileSpecObject;
            for (final TileSpec tileSpec : tileSpecs) {
                query = new Document("tileId", tileSpec.getTileId());
                tileSpecObject = Document.parse(tileSpec.toJson());
                modelList.add(new ReplaceOneModel<>(query, tileSpecObject, UPSERT_OPTION));
            }

            final BulkWriteResult result = tileCollection.bulkWrite(modelList, UNORDERED_OPTION);

            if (LOG.isDebugEnabled()) {
                final String bulkResultMessage = getBulkResultMessage("tile specs", result, tileSpecs.size());
                LOG.debug("saveResolvedTiles: {} using {}.initializeUnorderedBulkOp()",
                          bulkResultMessage, getFullName(tileCollection), query.toJson());
            }
        }

    }

    /**
     * Saves the specified tile spec to the database.
     *
     * @param  stackId    stack identifier.
     * @param  tileSpec   specification to be saved.
     *
     * @return the specification updated with any attributes that were modified by the save.
     *
     * @throws IllegalArgumentException
     *   if any required parameters or transform spec references are missing.
     */
    public TileSpec saveTileSpec(final StackId stackId,
                                 final TileSpec tileSpec)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("tileSpec", tileSpec);
        validateRequiredParameter("tileSpec.tileId", tileSpec.getTileId());

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        final String context = "tile spec with id '" + tileSpec.getTileId();
        validateTransformReferences(context, stackId, tileSpec.getTransforms());

        final Document query = new Document();
        query.put("tileId", tileSpec.getTileId());

        final Document tileSpecObject = Document.parse(tileSpec.toJson());

        final UpdateResult result = tileCollection.replaceOne(query, tileSpecObject, UPSERT_OPTION);

        LOG.debug("saveTileSpec: {}.{},({}), upsertedId is {}",
                  getFullName(tileCollection), getInsertOrUpdate(result), query.toJson(), result.getUpsertedId());

        return tileSpec;
    }

    /**
     * @return the specified transform spec.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing.
     *
     * @throws ObjectNotFoundException
     *   if a spec with the specified transformId cannot be found.
     */
    public TransformSpec getTransformSpec(final StackId stackId,
                                          final String transformId)
            throws IllegalArgumentException,
                   ObjectNotFoundException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("transformId", transformId);

        final MongoCollection<Document> transformCollection = getTransformCollection(stackId);

        final Document query = new Document();
        query.put("id", transformId);

        LOG.debug("getTransformSpec: {}.find({})", getFullName(transformCollection), query.toJson());

        final Document document = transformCollection.find(query).first();

        if (document == null) {
            throw new ObjectNotFoundException("transform spec with id '" + transformId + "' does not exist in the " +
                                              getFullName(transformCollection) + " collection");
        }

        return TransformSpec.fromJson(document.toJson());
    }

    /**
     * Saves the specified transform spec to the database.
     *
     * @param  stackId        stack identifier.
     * @param  transformSpec  specification to be saved.
     *
     * @return the specification updated with any attributes that were modified by the save.
     *
     * @throws IllegalArgumentException
     *   if any required parameters or transform spec references are missing.
     */
    public TransformSpec saveTransformSpec(final StackId stackId,
                                           final TransformSpec transformSpec)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("transformSpec", transformSpec);
        validateRequiredParameter("transformSpec.id", transformSpec.getId());

        final MongoCollection<Document> transformCollection = getTransformCollection(stackId);

        final String context = "transform spec with id '" + transformSpec.getId() + "'";
        validateTransformReferences(context, stackId, transformSpec);

        final Document query = new Document();
        query.put("id", transformSpec.getId());

        final Document transformSpecObject = Document.parse(transformSpec.toJson());

        final UpdateResult result = transformCollection.replaceOne(query, transformSpecObject, UPSERT_OPTION);

        LOG.debug("saveTransformSpec: {}.{},({}), upsertedId is {}",
                  getFullName(transformCollection), getInsertOrUpdate(result), query.toJson(), result.getUpsertedId());

        return transformSpec;
    }

    /**
     * @return list of distinct z values (layers) for the specified stackId.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public List<Double> getZValues(final StackId stackId)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        final List<Double> list = new ArrayList<>();
        for (final Double zValue : tileCollection.distinct("z", Double.class)) {
            if (zValue != null) {
                list.add(zValue);
            }
        }

        LOG.debug("getZValues: returning {} values for {}", list.size(), getFullName(tileCollection));

        return list;
    }

    public Double getZForSection(final StackId stackId,
                                 final String sectionId)
            throws IllegalArgumentException, IllegalStateException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("sectionId", sectionId);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final Document query = new Document("layout.sectionId", sectionId);

        final Document document = tileCollection.find(query).first();

        if (document == null) {
            throw new ObjectNotFoundException("sectionId '" + sectionId + "' does not exist in the " +
                                              getFullName(tileCollection) + " collection");
        }

        final TileSpec tileSpec = TileSpec.fromJson(document.toJson());

        return tileSpec.getZ();
    }

    public void updateZForSection(final StackId stackId,
                                  final String sectionId,
                                  final Double z)
            throws IllegalArgumentException, IllegalStateException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("sectionId", sectionId);
        validateRequiredParameter("z", z);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final Document query = new Document("layout.sectionId", sectionId);
        final Document update = new Document("$set", new Document("z", z));

        final UpdateResult result = tileCollection.updateMany(query, update);

        LOG.debug("updateZForSection: updated {} tile specs with {}.update({},{})",
                  result.getModifiedCount(), getFullName(tileCollection), query.toJson(), update.toJson());
    }


    /**
     * @return list of section data objects for the specified stackId.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public List<SectionData> getSectionData(final StackId stackId)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);

        final List<SectionData> list = new ArrayList<>();

        if (! collectionExists(stackId.getSectionCollectionName())) {
            throw new IllegalArgumentException("section data not aggregated for " + stackId +
                                               ", set stack state to COMPLETE to generate the aggregate collection");
        }

        final MongoCollection<Document> sectionCollection = getSectionCollection(stackId);

        final Document query = new Document();

        try (MongoCursor<Document> cursor = sectionCollection.find(query).iterator()) {
            Document document;
            Document resultId;
            String sectionId;
            Double z;
            while (cursor.hasNext()) {
                document = cursor.next();
                resultId = document.get("_id", Document.class);
                sectionId = resultId.get("sectionId", String.class);
                z = resultId.get("z", Double.class);
                if ((sectionId != null) && (z != null)) {
                    list.add(new SectionData(sectionId, z));
                }
            }
        }

        LOG.debug("getSectionData: returning {} values for {}.find({})",
                  list.size(), sectionCollection.getNamespace().getFullName());

        return list;
    }

    /**
     * @return list of stack owners.
     *
     * @throws IllegalArgumentException
     *   if required parameters are not specified.
     */
    public List<String> getOwners()
            throws IllegalArgumentException {

        final List<String> list = new ArrayList<>();

        final MongoCollection<Document> stackMetaDataCollection = getStackMetaDataCollection();
        for (final String owner : stackMetaDataCollection.distinct("stackId.owner", String.class)) {
            list.add(owner);
        }

        return list;
    }

    /**
     * @return list of stack meta data objects for the specified owner.
     *
     * @throws IllegalArgumentException
     *   if required parameters are not specified.
     */
    public List<StackMetaData> getStackMetaDataListForOwner(final String owner)
            throws IllegalArgumentException {

        validateRequiredParameter("owner", owner);

        final List<StackMetaData> list = new ArrayList<>();

        final MongoCollection<Document> stackMetaDataCollection = getStackMetaDataCollection();
        final Document query = new Document("stackId.owner", owner);
        final Document sortCriteria = new Document(
                "stackId.project", 1).append(
                "currentVersion.cycleNumber", 1).append(
                "currentVersion.cycleStepNumber", 1).append(
                "stackId.name", 1);

        try (MongoCursor<Document> cursor = stackMetaDataCollection.find(query).sort(sortCriteria).iterator()) {
            Document document;
            while (cursor.hasNext()) {
                document = cursor.next();
                list.add(StackMetaData.fromJson(document.toJson()));
            }
        }

        LOG.debug("getStackMetaDataListForOwner: returning {} values for {}.find({}).sort({})",
                  list.size(), stackMetaDataCollection.getNamespace().getFullName(),
                  query.toJson(), sortCriteria.toJson());

        return list;
    }

    /**
     * @return meta data for the specified stack or null if the stack cannot be found.
     *
     * @throws IllegalArgumentException
     *   if required parameters are not specified.
     */
    public StackMetaData getStackMetaData(final StackId stackId)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);

        StackMetaData stackMetaData = null;

        final MongoCollection<Document> stackMetaDataCollection = getStackMetaDataCollection();
        final Document query = getStackIdQuery(stackId);

        final Document document = stackMetaDataCollection.find(query).first();
        if (document != null) {
            stackMetaData = StackMetaData.fromJson(document.toJson());
        }

        return stackMetaData;
    }

    public void saveStackMetaData(final StackMetaData stackMetaData) {

        LOG.debug("saveStackMetaData: entry, stackMetaData={}", stackMetaData);

        validateRequiredParameter("stackMetaData", stackMetaData);

        final StackId stackId = stackMetaData.getStackId();
        final MongoCollection<Document> stackMetaDataCollection = getStackMetaDataCollection();

        final Document query = getStackIdQuery(stackId);
        final Document stackMetaDataObject = Document.parse(stackMetaData.toJson());
        final UpdateResult result = stackMetaDataCollection.replaceOne(query, stackMetaDataObject, UPSERT_OPTION);

        final String action;
        if (result.getMatchedCount() > 0) {
            action = "update";
        } else {
            action = "insert";
            ensureCoreTransformIndex(getTransformCollection(stackId));
            ensureCoreTileIndexes(getTileCollection(stackId));
        }

        LOG.debug("saveStackMetaData: {}.{}({})",
                  stackMetaDataCollection.getNamespace().getFullName(), action, query.toJson());
    }

    public StackMetaData ensureIndexesAndDeriveStats(final StackMetaData stackMetaData) {

        validateRequiredParameter("stackMetaData", stackMetaData);

        final StackId stackId = stackMetaData.getStackId();

        LOG.debug("ensureIndexesAndDeriveStats: entry, {}", stackId);

        final MongoCollection<Document> transformCollection = getTransformCollection(stackId);
        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        // should not be necessary, but okay to ensure core indexes just in case
        ensureCoreTransformIndex(transformCollection);
        ensureCoreTileIndexes(tileCollection);

        ensureSupplementaryTileIndexes(tileCollection);

        deriveSectionData(stackId);

        final List<Double> zValues = getZValues(stackId);
        final long sectionCount = zValues.size();

        long nonIntegralSectionCount = 0;
        double truncatedZ;
        for (final Double z : zValues) {
            truncatedZ = (double) z.intValue();
            if (z > truncatedZ) {
                nonIntegralSectionCount++;
            }
        }

        final long tileCount = tileCollection.count();
        LOG.debug("ensureIndexesAndDeriveStats: tileCount for {} is {}", stackId, tileCount);

        final long transformCount = transformCollection.count();
        LOG.debug("ensureIndexesAndDeriveStats: transformCount for {} is {}, deriving aggregate stats ...",
                  stackId, transformCount);

        // db.<stack_prefix>__tile.aggregate(
        //     [
        //         {
        //             "$project":  {
        //                 "z": "$z",
        //                 "minX": "$minX",
        //                 "minY": "$minY",
        //                 "maxX": "$maxX",
        //                 "maxY": "$maxY",
        //                 "width":  { "$subtract": [ "$maxX", "$minX" ] },
        //                 "height": { "$subtract": [ "$maxY", "$minY" ] }
        //             }
        //         },
        //         {
        //             "$group": {
        //                 "_id": "minAndMaxValues",
        //                 "stackMinX": { "$min": "$minX" },
        //                 "stackMinY": { "$min": "$minY" },
        //                 "stackMinZ": { "$min": "$z" },
        //                 "stackMaxX": { "$max": "$maxX" },
        //                 "stackMaxY": { "$max": "$maxY" },
        //                 "stackMaxZ": { "$max": "$z" } } } )
        //                 "stackMinTileWidth":  { "$min": "$width" },
        //                 "stackMaxTileWidth":  { "$max": "$width" },
        //                 "stackMinTileHeight": { "$min": "$height" },
        //                 "stackMaxTileHeight": { "$max": "$height" }
        //             }
        //         }
        //     ]
        // )

        final Document tileWidth = new Document("$subtract", buildBasicDBList(new String[] {"$maxX","$minX" }));
        final Document tileHeight = new Document("$subtract", buildBasicDBList(new String[] {"$maxY","$minY" }));
        final Document tileValues = new Document("z", "$z").append(
                "minX", "$minX").append("minY", "$minY").append("maxX", "$maxX").append("maxY", "$maxY").append(
                "width", tileWidth).append("height", tileHeight);

        final Document projectStage = new Document("$project", tileValues);

        final String minXKey = "stackMinX";
        final String minYKey = "stackMinY";
        final String minZKey = "stackMinZ";
        final String maxXKey = "stackMaxX";
        final String maxYKey = "stackMaxY";
        final String maxZKey = "stackMaxZ";
        final String minWidthKey = "stackMinTileWidth";
        final String maxWidthKey = "stackMaxTileWidth";
        final String minHeightKey = "stackMinTileHeight";
        final String maxHeightKey = "stackMaxTileHeight";

        final Document minAndMaxValues = new Document("_id", "minAndMaxValues");
        minAndMaxValues.append(minXKey, new Document(QueryOperators.MIN, "$minX"));
        minAndMaxValues.append(minYKey, new Document(QueryOperators.MIN, "$minY"));
        minAndMaxValues.append(minZKey, new Document(QueryOperators.MIN, "$z"));
        minAndMaxValues.append(maxXKey, new Document(QueryOperators.MAX, "$maxX"));
        minAndMaxValues.append(maxYKey, new Document(QueryOperators.MAX, "$maxY"));
        minAndMaxValues.append(maxZKey, new Document(QueryOperators.MAX, "$z"));
        minAndMaxValues.append(minWidthKey, new Document(QueryOperators.MIN, "$width"));
        minAndMaxValues.append(maxWidthKey, new Document(QueryOperators.MAX, "$width"));
        minAndMaxValues.append(minHeightKey, new Document(QueryOperators.MIN, "$height"));
        minAndMaxValues.append(maxHeightKey, new Document(QueryOperators.MAX, "$height"));

        final Document groupStage = new Document("$group", minAndMaxValues);

        final List<Document> pipeline = new ArrayList<>();
        pipeline.add(projectStage);
        pipeline.add(groupStage);


        // mongodb java 3.0 driver notes:
        // -- need to set cursor batchSize to prevent NPE from cursor creation
        final Document aggregateResult = tileCollection.aggregate(pipeline).batchSize(1).first();

        if (aggregateResult == null) {
            String cause = "";
            if (tileCollection.count() == 0) {
                cause = " because the stack has no tiles";
            }
            throw new IllegalStateException("Stack data aggregation returned no results" + cause + ".  " +
                                            "The aggregation query was " + getFullName(tileCollection) +
                                            ".aggregate(" + pipeline + ") .");
        }

        final Bounds stackBounds = new Bounds(getDoubleValue(aggregateResult.get(minXKey)),
                                              getDoubleValue(aggregateResult.get(minYKey)),
                                              getDoubleValue(aggregateResult.get(minZKey)),
                                              getDoubleValue(aggregateResult.get(maxXKey)),
                                              getDoubleValue(aggregateResult.get(maxYKey)),
                                              getDoubleValue(aggregateResult.get(maxZKey)));

        final Integer minTileWidth = getDoubleValueAsInteger(aggregateResult.get(minWidthKey));
        final Integer maxTileWidth = getDoubleValueAsInteger(aggregateResult.get(maxWidthKey));
        final Integer minTileHeight = getDoubleValueAsInteger(aggregateResult.get(minHeightKey));
        final Integer maxTileHeight = getDoubleValueAsInteger(aggregateResult.get(maxHeightKey));

        final StackStats stats = new StackStats(stackBounds,
                                                sectionCount,
                                                nonIntegralSectionCount,
                                                tileCount,
                                                transformCount,
                                                minTileWidth,
                                                maxTileWidth,
                                                minTileHeight,
                                                maxTileHeight);
        stackMetaData.setStats(stats);

        LOG.debug("ensureIndexesAndDeriveStats: completed stat derivation for {}, stats={}", stackId, stats);

        stackMetaData.setState(StackMetaData.StackState.COMPLETE);

        final MongoCollection<Document> stackMetaDataCollection = getStackMetaDataCollection();
        final Document query = getStackIdQuery(stackId);
        final Document stackMetaDataObject = Document.parse(stackMetaData.toJson());
        final UpdateResult result = stackMetaDataCollection.replaceOne(query, stackMetaDataObject, UPSERT_OPTION);

        LOG.debug("ensureIndexesAndDeriveStats: {}.{}({})",
                  getFullName(stackMetaDataCollection), getInsertOrUpdate(result), query.toJson());

        return stackMetaData;
    }

    private void deriveSectionData(final StackId stackId)
            throws IllegalArgumentException {

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final String sectionCollectionName = stackId.getSectionCollectionName();

        // db.<stack_prefix>__tile.aggregate(
        //     [
        //         { "$group": { "_id": { "sectionId": "$layout.sectionId", "z": "$z" } } },
        //         { "$sort": { "_id.sectionId": 1 } }
        //     ]
        // )

        final Document idComponents = new Document("sectionId", "$layout.sectionId").append("z", "$z");
        final Document id = new Document("_id", idComponents);

        final List<Document> pipeline = new ArrayList<>();
        pipeline.add(new Document("$group", id));
        pipeline.add(new Document("$sort", new Document("_id.sectionId", 1)));
        pipeline.add(new Document("$out", sectionCollectionName));

        if (LOG.isDebugEnabled()) {
            LOG.debug("deriveSectionData: running {}.aggregate({})",
                      getFullName(tileCollection),
                      getJsonString(pipeline));
        }

        // mongodb java 3.0 driver notes:
        // -- need to retrieve first batch from cursor - via first() call - to force aggregate operation to run
        //    (even though we don't need to work with the batch result here)
        // -- need to set cursor batchSize to prevent NPE from cursor creation
        tileCollection.aggregate(pipeline).batchSize(0).first();

        if (! collectionExists(sectionCollectionName)) {
            throw new IllegalStateException("Section data aggregation results were not saved to " +
                                            sectionCollectionName);
        }

        final MongoCollection<Document> sectionCollection = getSectionCollection(stackId);
        final long sectionCount = sectionCollection.count();

        LOG.debug("deriveSectionData: saved data for {} sections in {}", sectionCount, getFullName(sectionCollection));
    }

    public void removeStack(final StackId stackId,
                            final boolean includeMetaData)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final long tileCount = tileCollection.count();
        tileCollection.drop();

        LOG.debug("removeStack: {}.drop() deleted {} document(s)", getFullName(tileCollection), tileCount);

        final MongoCollection<Document> transformCollection = getTransformCollection(stackId);
        final long transformCount = transformCollection.count();
        transformCollection.drop();

        LOG.debug("removeStack: {}.drop() deleted {} document(s)", getFullName(transformCollection), transformCount);

        if (includeMetaData) {
            final MongoCollection<Document> stackMetaDataCollection = getStackMetaDataCollection();
            final Document stackIdQuery = getStackIdQuery(stackId);
            final DeleteResult stackMetaDataRemoveResult = stackMetaDataCollection.deleteOne(stackIdQuery);

            LOG.debug("removeStack: {}.remove({}) deleted {} document(s)",
                      getFullName(stackMetaDataCollection),
                      stackIdQuery.toJson(),
                      stackMetaDataRemoveResult.getDeletedCount());
        }
    }

    public void removeSection(final StackId stackId,
                              final Double z)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final Document tileQuery = new Document("z", z);
        final DeleteResult removeResult = tileCollection.deleteMany(tileQuery);

        LOG.debug("removeSection: {}.remove({}) deleted {} document(s)",
                  getFullName(tileCollection), tileQuery.toJson(), removeResult.getDeletedCount());
    }

    /**
     * @return coordinate bounds for all tiles in the specified stack layer.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public Bounds getLayerBounds(final StackId stackId,
                                 final Double z)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);
        final Document tileQuery = new Document("z", z);

        final Double minX = getBound(tileCollection, tileQuery, "minX", true);

        if (minX == null) {
            throw new IllegalArgumentException("stack " + stackId.getStack() +
                                               " does not contain any tiles with a z value of " + z);
        }

        final Double minY = getBound(tileCollection, tileQuery, "minY", true);
        final Double maxX = getBound(tileCollection, tileQuery, "maxX", false);
        final Double maxY = getBound(tileCollection, tileQuery, "maxY", false);

        return new Bounds(minX, minY, z, maxX, maxY, z);
    }

    /**
     * @return spatial data for all tiles in the specified stack layer.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     */
    public List<TileBounds> getTileBounds(final StackId stackId,
                                          final Double z)
            throws IllegalArgumentException {

        validateRequiredParameter("stackId", stackId);
        validateRequiredParameter("z", z);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        // EXAMPLE:   find({"z" : 3466.0},{"tileId": 1, "minX": 1, "minY": 1, "maxX": 1, "maxY": 1, "_id": 0})
        // INDEX:     z_1_minY_1_minX_1_maxY_1_maxX_1_tileId_1
        final Document tileQuery = new Document("z", z);
        final Document tileKeys =
                new Document("tileId", 1).append(
                        "minX", 1).append("minY", 1).append("maxX", 1).append("maxY", 1).append("_id", 0);

        final List<TileBounds> list = new ArrayList<>();

        try (MongoCursor<Document> cursor = tileCollection.find(tileQuery).projection(tileKeys).iterator()) {
            Document document;
            while (cursor.hasNext()) {
                document = cursor.next();
                list.add(TileBounds.fromJson(document.toJson()));
            }
        }

        LOG.debug("getTileBounds: found {} tile spec(s) for {}.find({},{})",
                  list.size(), getFullName(tileCollection), tileQuery.toJson(), tileKeys.toJson());

        return list;
    }

    public void cloneStack(final StackId fromStackId,
                           final StackId toStackId,
                           final List<Double> zValues)
            throws IllegalArgumentException, IllegalStateException {

        validateRequiredParameter("fromStackId", fromStackId);
        validateRequiredParameter("toStackId", toStackId);

        final MongoCollection<Document> fromTransformCollection = getTransformCollection(fromStackId);
        final MongoCollection<Document> toTransformCollection = getTransformCollection(toStackId);
        cloneCollection(fromTransformCollection, toTransformCollection, new Document());

        final Document filterQuery = new Document();
        if ((zValues != null) && (zValues.size() > 0)) {
            final BasicDBList list = new BasicDBList();
            list.addAll(zValues);
            final Document zFilter = new Document(QueryOperators.IN, list);
            filterQuery.append("z", zFilter);
        }

        final MongoCollection<Document> fromTileCollection = getTileCollection(fromStackId);
        final MongoCollection<Document> toTileCollection = getTileCollection(toStackId);
        cloneCollection(fromTileCollection, toTileCollection, filterQuery);
    }

    /**
     * Writes the layout file data for the specified stack to the specified stream.
     *
     * @param  stackId          stack identifier.
     * @param  stackRequestUri  the base stack request URI for building tile render-parameter URIs.
     * @param  minZ             the minimum z to include (or null if no minimum).
     * @param  maxZ             the maximum z to include (or null if no maximum).
     * @param  outputStream     stream to which layout file data is to be written.
     *
     * @throws IllegalArgumentException
     *   if any required parameters are missing or the stack cannot be found.
     *
     * @throws IOException
     *   if the data cannot be written for any reason.
     */
    public void writeLayoutFileData(final StackId stackId,
                                    final String stackRequestUri,
                                    final Double minZ,
                                    final Double maxZ,
                                    final OutputStream outputStream)
            throws IllegalArgumentException, IOException {

        LOG.debug("writeLayoutFileData: entry, stackId={}, minZ={}, maxZ={}",
                  stackId, minZ, maxZ);

        validateRequiredParameter("stackId", stackId);

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        Document zFilter = null;
        if (minZ != null) {
            zFilter = new Document(QueryOperators.GTE, minZ);
            if (maxZ != null) {
                zFilter = zFilter.append(QueryOperators.LTE, maxZ);
            }
        } else if (maxZ != null) {
            zFilter = new Document(QueryOperators.LTE, maxZ);
        }

        // EXAMPLE:   find({"z": {"$gte": 4370.0, "$lte": 4370.0}}, {"tileId": 1, "z": 1, "minX": 1, "minY": 1, "layout": 1, "mipmapLevels": 1}).sort({"z": 1, "minY": 1, "minX": 1})
        // INDEX:     z_1_minY_1_minX_1_maxY_1_maxX_1_tileId_1

        final Document tileQuery;
        if (zFilter == null) {
            tileQuery = new Document();
        } else {
            tileQuery = new Document("z", zFilter);
        }

        final Document tileKeys = new Document();

        final ProcessTimer timer = new ProcessTimer();
        int tileSpecCount = 0;
        final Document orderBy = new Document("z", 1).append("minY", 1).append("minX", 1);
        try (MongoCursor<Document> cursor = tileCollection.find(tileQuery).projection(tileKeys).sort(orderBy).iterator()) {
            final String baseUriString = '\t' + stackRequestUri + "/tile/";

            Document document;
            TileSpec tileSpec;
            String layoutData;
            String uriString;
            while (cursor.hasNext()) {
                document = cursor.next();
                tileSpec = TileSpec.fromJson(document.toJson());
                layoutData = tileSpec.toLayoutFileFormat();
                outputStream.write(layoutData.getBytes());

                // {stackRequestUri}/tile/{tileId}/render-parameters
                uriString = baseUriString + tileSpec.getTileId() + "/render-parameters" + "\n";
                outputStream.write(uriString.getBytes());
                tileSpecCount++;

                if (timer.hasIntervalPassed()) {
                    LOG.debug("writeLayoutFileData: data written for {} tiles", tileSpecCount);
                }

            }
        }

        LOG.debug("writeLayoutFileData: wrote data for {} tile spec(s) returned by {}.find({},{}).sort({}), elapsedSeconds={}",
                  tileSpecCount, getFullName(tileCollection),
                  tileQuery.toJson(), tileKeys.toJson(), orderBy.toJson(), timer.getElapsedSeconds());
    }

    private List<TransformSpec> getTransformSpecs(final MongoCollection<Document> transformCollection,
                                                  final Set<String> specIds) {
        final int specCount = specIds.size();
        final List<TransformSpec> transformSpecList = new ArrayList<>(specCount);
        if (specCount > 0) {

            final Document transformQuery = new Document();
            transformQuery.put("id", new Document(QueryOperators.IN, specIds));

            LOG.debug("getTransformSpecs: {}.find({})", getFullName(transformCollection), transformQuery.toJson());

            try (MongoCursor<Document> cursor = transformCollection.find(transformQuery).iterator()) {
                Document document;
                TransformSpec transformSpec;
                while (cursor.hasNext()) {
                    document = cursor.next();
                    transformSpec = TransformSpec.fromJson(document.toJson());
                    transformSpecList.add(transformSpec);
                }
            }

        }

        return transformSpecList;
    }

    private void getDataForTransformSpecReferences(final MongoCollection<Document> transformCollection,
                                                   final Set<String> unresolvedSpecIds,
                                                   final Map<String, TransformSpec> resolvedIdToSpecMap,
                                                   final int callCount) {

        if (callCount > 10) {
            throw new IllegalStateException(callCount + " passes have been made to resolve transform references, " +
                                            "exiting in case the data is overly nested or there is a recursion error");
        }

        final int specCount = unresolvedSpecIds.size();
        if (specCount > 0) {

            final List<TransformSpec> transformSpecList = getTransformSpecs(transformCollection,
                                                                            unresolvedSpecIds);

            LOG.debug("resolveTransformSpecReferences: on pass {} retrieved {} transform specs",
                      callCount, transformSpecList.size());

            final Set<String> newlyUnresolvedSpecIds = new HashSet<>();

            for (final TransformSpec spec : transformSpecList) {
                resolvedIdToSpecMap.put(spec.getId(), spec);
                for (final String id : spec.getUnresolvedIds()) {
                    if ((! resolvedIdToSpecMap.containsKey(id)) && (! unresolvedSpecIds.contains(id))) {
                        newlyUnresolvedSpecIds.add(id);
                    }
                }
            }

            if (newlyUnresolvedSpecIds.size() > 0) {
                getDataForTransformSpecReferences(transformCollection,
                                                  newlyUnresolvedSpecIds,
                                                  resolvedIdToSpecMap,
                                                  (callCount + 1));
            }
        }
    }

    private Map<String, TransformSpec> addResolvedTileSpecs(final StackId stackId,
                                                            final Document tileQuery,
                                                            final RenderParameters renderParameters) {

        final MongoCollection<Document> tileCollection = getTileCollection(stackId);

        // EXAMPLE:   find({"z": 4050.0 , "minX": {"$lte": 239850.0} , "minY": {"$lte": 149074.0}, "maxX": {"$gte": -109.0}, "maxY": {"$gte": 370.0}}).sort({"tileId": 1})
        // INDEXES:   z_1_minY_1_minX_1_maxY_1_maxX_1_tileId_1 (z1_minX_1, z1_maxX_1, ... used for edge cases)

        // order tile specs by tileId to ensure consistent coordinate mapping
        final Document orderBy = new Document("tileId", 1);

        try (MongoCursor<Document> cursor = tileCollection.find(tileQuery).sort(orderBy).iterator()) {
            Document document;
            TileSpec tileSpec;
            int count = 0;
            while (cursor.hasNext()) {
                if (count > 50000) {
                    throw new IllegalArgumentException("query too broad, over " + count + " tiles match " + tileQuery);
                }
                document = cursor.next();
                tileSpec = TileSpec.fromJson(document.toJson());
                renderParameters.addTileSpec(tileSpec);
                count++;
            }
        }

        LOG.debug("addResolvedTileSpecs: found {} tile spec(s) for {}.find({}).sort({})",
                  renderParameters.numberOfTileSpecs(), getFullName(tileCollection),
                  tileQuery.toJson(), orderBy.toJson());

        return resolveTransformReferencesForTiles(stackId, renderParameters.getTileSpecs());
    }

    private Document lte(final double value) {
        return new Document(QueryOperators.LTE, value);
    }

    private Document gte(final double value) {
        return new Document(QueryOperators.GTE, value);
    }

    private Document getIntersectsBoxQuery(final double z,
                                           final double x,
                                           final double y,
                                           final double lowerRightX,
                                           final double lowerRightY) {
        // intersection logic stolen from java.awt.Rectangle#intersects (without overflow checks)
        //   rx => minX, ry => minY, rw => maxX,        rh => maxY
        //   tx => x,    ty => y,    tw => lowerRightX, th => lowerRightY
//        return Filters.eq("z", z)
        return new Document("z", z).append(
                "minX", lte(lowerRightX)).append(
                "minY", lte(lowerRightY)).append(
                "maxX", gte(x)).append(
                "maxY", gte(y));
    }

    private Document getStackIdQuery(final StackId stackId) {
        return new Document(
                "stackId.owner", stackId.getOwner()).append(
                "stackId.project", stackId.getProject()).append(
                "stackId.stack", stackId.getStack());
    }

    private Document getGroupQuery(final Double minZ,
                                   final Double maxZ,
                                   final String groupId,
                                   final Double minX,
                                   final Double maxX,
                                   final Double minY,
                                   final Double maxY)
            throws IllegalArgumentException {

        final Document groupQuery = new Document();

        if ((minZ != null) && minZ.equals(maxZ)) {
            groupQuery.append("z", minZ);
        } else {
            if (minZ != null) {
                groupQuery.append("z", new Document(QueryOperators.GTE, minZ));
            }
            if (maxZ != null) {
                groupQuery.append("z", new Document(QueryOperators.LTE, maxZ));
            }
        }

        if (groupId != null) {
            groupQuery.append("groupId", groupId);
        }

        if (minX != null) {
            groupQuery.append("maxX", new Document(QueryOperators.GTE, minX));
        }
        if (maxX != null) {
            groupQuery.append("minX", new Document(QueryOperators.LTE, maxX));
        }
        if (minY != null) {
            groupQuery.append("maxY", new Document(QueryOperators.GTE, minY));
        }
        if (maxY != null) {
            groupQuery.append("minY", new Document(QueryOperators.LTE, maxY));
        }

        return groupQuery;
    }

    private BasicDBList buildBasicDBList(final String[] values) {
        final BasicDBList list = new BasicDBList();
        Collections.addAll(list, values);
        return list;
    }

    private void validateTransformReferences(final String context,
                                             final StackId stackId,
                                             final TransformSpec transformSpec) {

        final Set<String> unresolvedTransformSpecIds = transformSpec.getUnresolvedIds();

        if (unresolvedTransformSpecIds.size() > 0) {
            final MongoCollection<Document> transformCollection = getTransformCollection(stackId);
            final List<TransformSpec> transformSpecList = getTransformSpecs(transformCollection,
                                                                            unresolvedTransformSpecIds);
            if (transformSpecList.size() != unresolvedTransformSpecIds.size()) {
                final Set<String> existingIds = new HashSet<>(transformSpecList.size());
                for (final TransformSpec existingTransformSpec : transformSpecList) {
                    existingIds.add(existingTransformSpec.getId());
                }
                final Set<String> missingIds = new TreeSet<>();
                for (final String id : unresolvedTransformSpecIds) {
                    if (! existingIds.contains(id)) {
                        missingIds.add(id);
                    }
                }
                throw new IllegalArgumentException(context + " references transform id(s) " + missingIds +
                                                   " which do not exist in the " +
                                                   getFullName(transformCollection) + " collection");
            }
        }
    }

    private Double getBound(final MongoCollection<Document> tileCollection,
                            final Document tileQuery,
                            final String boundKey,
                            final boolean isMin) {

        Double bound = null;

        final Document query = new Document(tileQuery);

        // Add a $gt / $lt constraint to ensure that null values aren't included and
        // that an indexOnly query is possible ($exists is not sufficient).
        int order = -1;
        if (isMin) {
            order = 1;
            // Double.MIN_VALUE is the minimum positive double value, so we need to use -Double.MAX_VALUE here
            query.append(boundKey, new Document("$gt", -Double.MAX_VALUE));
        } else {
            query.append(boundKey, new Document("$lt", Double.MAX_VALUE));
        }

        final Document tileKeys = new Document(boundKey, 1).append("_id", 0);
        final Document orderBy = new Document(boundKey, order);

        // EXAMPLE:   find({ "z" : 3299.0},{ "minX" : 1 , "_id" : 0}).sort({ "minX" : 1}).limit(1)
        // INDEXES:   z_1_minX_1, z_1_minY_1, z_1_maxX_1, z_1_maxY_1
        final Document document = tileCollection.find(query).projection(tileKeys).sort(orderBy).first();
        if (document != null) {
            bound = document.get(boundKey, Double.class);
        }

        LOG.debug("getBound: returning {} for {}.find({},{}).sort({}).limit(1)",
                  bound, getFullName(tileCollection), query.toJson(), tileKeys.toJson(), orderBy.toJson());

        return bound;
    }

    private void cloneCollection(final MongoCollection<Document> fromCollection,
                                 final MongoCollection<Document> toCollection,
                                 final Document filterQuery)
            throws IllegalStateException {

        final long fromCount = fromCollection.count();
        final long toCount;
        final String fromFullName = getFullName(fromCollection);
        final String toFullName = getFullName(toCollection);

        LOG.debug("cloneCollection: entry, copying up to {} documents from {} to {}",
                  fromCount, fromFullName, toFullName);

        final ProcessTimer timer = new ProcessTimer(15000);

        // We use bulk inserts to improve performance, but we still need to chunk
        // the bulk operations to avoid memory issues with large collections.
        // This 10,000 document chunk size is arbitrary but seems to be sufficient.
        final int maxDocumentsPerBulkInsert = 10000;

        long count = 0;

        final List<WriteModel<Document>> modelList = new ArrayList<>(maxDocumentsPerBulkInsert);

        try (MongoCursor<Document> cursor = fromCollection.find(filterQuery).iterator()) {

            BulkWriteResult result;
            long insertedCount;
            Document document;
            while (cursor.hasNext()) {
                document = cursor.next();
                modelList.add(new InsertOneModel<>(document));
                count++;
                if (count % maxDocumentsPerBulkInsert == 0) {
                    result = toCollection.bulkWrite(modelList, UNORDERED_OPTION);
                    insertedCount = result.getInsertedCount();
                    if (insertedCount != maxDocumentsPerBulkInsert) {
                        throw new IllegalStateException("only inserted " + insertedCount + " out of " +
                                                        maxDocumentsPerBulkInsert +
                                                        " documents for batch ending with document " + count);
                    }
                    modelList.clear();
                    if (timer.hasIntervalPassed()) {
                        LOG.debug("cloneCollection: inserted {} documents", count);
                    }
                }
            }

            if (count % maxDocumentsPerBulkInsert > 0) {
                result = toCollection.bulkWrite(modelList, UNORDERED_OPTION);
                insertedCount = result.getInsertedCount();
                final long expectedCount = count % maxDocumentsPerBulkInsert;
                if (insertedCount != expectedCount) {
                    throw new IllegalStateException("only inserted " + insertedCount + " out of " +
                                                    expectedCount +
                                                    " documents for last batch ending with document " + count);
                }
            }

            toCount = toCollection.count();

            // if nothing was filtered, verify that all documents got copied
            if (filterQuery.keySet().size() == 0) {
                if (toCount != fromCount) {
                    throw new IllegalStateException("only inserted " + toCount + " out of " + fromCount + " documents");
                }
            }

        }

        LOG.debug("cloneCollection: inserted {} documents from {}.find(\\{}) to {}",
                  toCount, fromFullName, toFullName);
    }

    private MongoCollection<Document> getStackMetaDataCollection() {
        return renderDatabase.getCollection(STACK_META_DATA_COLLECTION_NAME);
    }

    private MongoCollection<Document> getTileCollection(final StackId stackId) {
        return renderDatabase.getCollection(stackId.getTileCollectionName());
    }

    private MongoCollection<Document> getSectionCollection(final StackId stackId) {
        return renderDatabase.getCollection(stackId.getSectionCollectionName());
    }

    private MongoCollection<Document> getTransformCollection(final StackId stackId) {
        return renderDatabase.getCollection(stackId.getTransformCollectionName());
    }

    private void validateRequiredParameter(final String context,
                                           final Object value)
            throws IllegalArgumentException {

        if (value == null) {
            throw new IllegalArgumentException(context + " value must be specified");
        }
    }

    private void ensureCoreTransformIndex(final MongoCollection<Document> transformCollection) {
        createIndex(transformCollection,
                    new Document("id", 1),
                    new IndexOptions().unique(true).background(true));
        LOG.debug("ensureCoreTransformIndex: exit");
    }

    private void ensureCoreTileIndexes(final MongoCollection<Document> tileCollection) {
        createIndex(tileCollection,
                    new Document("tileId", 1),
                    new IndexOptions().unique(true).background(true));
        createIndex(tileCollection, new Document("z", 1), BACKGROUND_OPTION);
        LOG.debug("ensureCoreTileIndex: exit");
    }

    private void ensureSupplementaryTileIndexes(final MongoCollection<Document> tileCollection) {

        // compound index used for bulk (e.g. resolvedTile) queries that sort by tileId
        createIndex(tileCollection, new Document("z", 1).append("tileId", 1), BACKGROUND_OPTION);

        createIndex(tileCollection, new Document("z", 1).append("minX", 1), BACKGROUND_OPTION);
        createIndex(tileCollection, new Document("z", 1).append("minY", 1), BACKGROUND_OPTION);
        createIndex(tileCollection, new Document("z", 1).append("maxX", 1), BACKGROUND_OPTION);
        createIndex(tileCollection, new Document("z", 1).append("maxY", 1), BACKGROUND_OPTION);

        // compound index used for most box intersection queries
        // - z, minY, minX order used to match layout file sorting needs
        // - appended tileId so that getTileBounds query can be index only (must not sort)
        createIndex(tileCollection,
                    new Document("z", 1).append(
                            "minY", 1).append("minX", 1).append("maxY", 1).append("maxX", 1).append("tileId", 1),
                    BACKGROUND_OPTION);

        // compound index used for group queries
        createIndex(tileCollection,
                    new Document("z", 1).append(
                            "groupId", 1).append("minY", 1).append("minX", 1).append("maxY", 1).append("maxX", 1),
                    BACKGROUND_OPTION);

        LOG.debug("ensureSupplementaryTileIndexes: exit");
    }

    private void createIndex(final MongoCollection<Document> collection,
                             final Document keys,
                             final IndexOptions options) {
        LOG.debug("createIndex: entry, collection={}, keys={}, options={}",
                  getFullName(collection), keys.toJson(), getOptionsJson(options));
        collection.createIndex(keys, options);
    }

    private boolean collectionExists(final String collectionName) {
        for (final String name : renderDatabase.listCollectionNames()) {
            if (name.equalsIgnoreCase(collectionName)) {
                return true;
            }
        }
        return false;
    }

    private String getFullName(final MongoCollection<Document> collection) {
        return collection.getNamespace().getFullName();
    }

    private String getJsonString(final List<Document> list) {
        final StringBuilder sb = new StringBuilder(1024);
        sb.append('[');
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(list.get(i).toJson());
        }
        sb.append(']');
        return sb.toString();
    }

    private String getBulkResultMessage(final String context,
                                        final BulkWriteResult result,
                                        final int objectCount) {

        final StringBuilder message = new StringBuilder(128);

        message.append("processed ").append(objectCount).append(" ").append(context);

        if (result.wasAcknowledged()) {
            final int updates = result.getMatchedCount();
            final int inserts = objectCount - updates;
            message.append(" with ").append(inserts).append(" inserts and ");
            message.append(updates).append(" updates");
        } else {
            message.append(" (result NOT acknowledged)");
        }

        return message.toString();
    }

    private Integer getDoubleValueAsInteger(final Object value) {
        Integer integerValue = null;
        final Double doubleValue = getDoubleValue(value);
        if (doubleValue != null) {
            integerValue = doubleValue.intValue();
        }
        return integerValue;
    }

    private Double getDoubleValue(final Object value) {
        Double doubleValue = null;
        if (value != null) {
            doubleValue = new Double(value.toString());
        }
        return doubleValue;
    }

    @Nonnull
    private String getInsertOrUpdate(final UpdateResult result) {
        final String action;
        if (result.getMatchedCount() > 0) {
            action = "update";
        } else {
            action = "insert";
        }
        return action;
    }

    private String getOptionsJson(final IndexOptions options) {

        @SuppressWarnings("MismatchedQueryAndUpdateOfCollection")
        final Document document = new Document();

        if (options.isBackground()) {
            document.append("background", true);
        }
        if (options.isUnique()) {
            document.append("unique", true);
        }

        // add other options if/when they are used

        return document.toJson();
    }

    private static final Logger LOG = LoggerFactory.getLogger(RenderDao.class);

    private static final IndexOptions BACKGROUND_OPTION = new IndexOptions().background(true);
    private static final UpdateOptions UPSERT_OPTION = new UpdateOptions().upsert(true);
    private static final BulkWriteOptions UNORDERED_OPTION = new BulkWriteOptions().ordered(false);
}
