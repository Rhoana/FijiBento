package org.janelia.render.service;

import java.io.IOException;
import java.io.OutputStream;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.WebApplicationException;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.StreamingOutput;
import javax.ws.rs.core.UriInfo;

import org.janelia.alignment.ImageAndMask;
import org.janelia.alignment.RenderParameters;
import org.janelia.alignment.spec.Bounds;
import org.janelia.alignment.spec.ResolvedTileSpecCollection;
import org.janelia.alignment.spec.SectionData;
import org.janelia.alignment.spec.TileBounds;
import org.janelia.alignment.spec.TileSpec;
import org.janelia.alignment.spec.TransformSpec;
import org.janelia.alignment.spec.stack.MipmapPathBuilder;
import org.janelia.alignment.spec.stack.StackId;
import org.janelia.alignment.spec.stack.StackMetaData;
import org.janelia.render.service.dao.RenderDao;
import org.janelia.render.service.model.IllegalServiceArgumentException;
import org.janelia.render.service.util.RenderServiceUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;

import static org.janelia.alignment.spec.stack.StackMetaData.StackState.LOADING;

/**
 * APIs for accessing tile and transform data stored in the Render service database.
 *
 * @author Eric Trautman
 */
@Path("/v1/owner/{owner}")
@Api(tags = {"Render Data APIs"},
     description = "Render Data Service")
public class RenderDataService {

    private final RenderDao renderDao;

    @SuppressWarnings("UnusedDeclaration")
    public RenderDataService()
            throws UnknownHostException {
        this(RenderDao.build());
    }

    public RenderDataService(final RenderDao renderDao) {
        this.renderDao = renderDao;
    }

    @Path("project/{project}/stack/{stack}/layoutFile")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    @ApiOperation(
            value = "Get layout file text for all stack layers",
            produces = MediaType.TEXT_PLAIN)
    public Response getLayoutFile(@PathParam("owner") final String owner,
                                  @PathParam("project") final String project,
                                  @PathParam("stack") final String stack,
                                  @Context final UriInfo uriInfo) {

        return getLayoutFileForZRange(owner, project, stack, null, null, uriInfo);
    }

    @Path("project/{project}/stack/{stack}/zRange/{minZ},{maxZ}/layoutFile")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    @ApiOperation(
            value = "Get layout file text for specified stack layers",
            produces = MediaType.TEXT_PLAIN)
    public Response getLayoutFileForZRange(@PathParam("owner") final String owner,
                                           @PathParam("project") final String project,
                                           @PathParam("stack") final String stack,
                                           @PathParam("minZ") final Double minZ,
                                           @PathParam("maxZ") final Double maxZ,
                                           @Context final UriInfo uriInfo) {

        LOG.info("getLayoutFileForZRange: entry, owner={}, project={}, stack={}, minZ={}, maxZ={}",
                 owner, project, stack, minZ, maxZ);

        Response response = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);

            final String requestUri = uriInfo.getRequestUri().toString();
            final String stackUri = "/stack/" + stack + "/";
            final int stackEnd = requestUri.indexOf(stackUri) + stackUri.length() - 1;
            final String stackRequestUri = requestUri.substring(0, stackEnd);
            final StreamingOutput responseOutput = new StreamingOutput() {
                @Override
                public void write(final OutputStream output)
                        throws IOException, WebApplicationException {
                    renderDao.writeLayoutFileData(stackId, stackRequestUri, minZ, maxZ, output);
                }
            };
            response = Response.ok(responseOutput).build();
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        return response;
    }

    @Path("project/{project}/stack/{stack}/zValues")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<Double> getZValues(@PathParam("owner") final String owner,
                                   @PathParam("project") final String project,
                                   @PathParam("stack") final String stack) {

        LOG.info("getZValues: entry, owner={}, project={}, stack={}",
                 owner, project, stack);

        List<Double> list = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            list = renderDao.getZValues(stackId);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return list;
    }

    @Path("project/{project}/stack/{stack}/sectionData")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<SectionData> getSectionData(@PathParam("owner") final String owner,
                                            @PathParam("project") final String project,
                                            @PathParam("stack") final String stack) {

        LOG.info("getSectionData: entry, owner={}, project={}, stack={}",
                 owner, project, stack);

        List<SectionData> list = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            list = renderDao.getSectionData(stackId);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return list;
    }

    @Path("project/{project}/stack/{stack}/reorderedSectionData")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<SectionData> getReorderedSectionData(@PathParam("owner") final String owner,
                                                     @PathParam("project") final String project,
                                                     @PathParam("stack") final String stack) {

        LOG.info("getReorderedSectionData: entry, owner={}, project={}, stack={}",
                 owner, project, stack);

        final List<SectionData> list = getSectionData(owner, project, stack);
        final List<SectionData> filteredList = new ArrayList<>(list.size());
        int sectionIdInt;
        int zInt;
        for (final SectionData sectionData : list) {
            try {
                sectionIdInt = (int) Double.parseDouble(sectionData.getSectionId());
                zInt = sectionData.getZ().intValue();
            } catch (final Exception e) {
                throw new IllegalServiceArgumentException(
                        "reordered sections cannot be determined because " +
                        "stack contains non-standard sectionId (" + sectionData.getSectionId() +
                        ") or z value (" + sectionData.getZ() + ")", e);
            }
            if (sectionIdInt != zInt) {
                filteredList.add(sectionData);
            }
        }
        return filteredList;
    }

    @Path("project/{project}/stack/{stack}/mergedZValues")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<Double> getMergedZValues(@PathParam("owner") final String owner,
                                         @PathParam("project") final String project,
                                         @PathParam("stack") final String stack) {

        LOG.info("getMergedZValues: entry, owner={}, project={}, stack={}",
                 owner, project, stack);

        final List<SectionData> sectionDataList = getSectionData(owner, project, stack);
        final Map<Double, String> zToSectionIdMap = new HashMap<>(sectionDataList.size() * 2);
        final Set<Double> mergedZValues = new HashSet<>();

        String previousSectionIdForZ;
        for (final SectionData sectionData : sectionDataList) {
            previousSectionIdForZ = zToSectionIdMap.put(sectionData.getZ(), sectionData.getSectionId());
            if (previousSectionIdForZ != null) {
                mergedZValues.add(sectionData.getZ());
            }
        }

        final List<Double> sortedZList = new ArrayList<>(mergedZValues);
        Collections.sort(sortedZList);

        return sortedZList;
    }

    @Path("project/{project}/stack/{stack}/highDoseLowDoseZValues")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<Double> getHighDoseLowDoseZValues(@PathParam("owner") final String owner,
                                                  @PathParam("project") final String project,
                                                  @PathParam("stack") final String stack) {

        final List<Double> list = getZValues(owner, project, stack);
        final LinkedHashSet<Double> filteredSet = new LinkedHashSet<>(list.size());
        Double lastHighDoseZ = -1.0;
        for (final Double z : list) {
            if ((z - z.intValue()) > 0) {
                if (z.intValue() == lastHighDoseZ.intValue()) {
                    filteredSet.add(lastHighDoseZ);
                    filteredSet.add(z);
                } else {
                    LOG.warn("getHighDoseLowDoseZValues: z {} is missing corresponding high dose (.0) section", z);
                }
            } else {
                lastHighDoseZ = z;
            }
        }

        LOG.info("getHighDoseLowDoseZValues: returning {} values", filteredSet.size());

        return new ArrayList<>(filteredSet);
    }

    @Path("project/{project}/stack/{stack}/z/{z}/bounds")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Bounds getLayerBounds(@PathParam("owner") final String owner,
                                 @PathParam("project") final String project,
                                 @PathParam("stack") final String stack,
                                 @PathParam("z") final Double z) {

        LOG.info("getLayerBounds: entry, owner={}, project={}, stack={}, z={}",
                 owner, project, stack, z);

        Bounds bounds = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            bounds = renderDao.getLayerBounds(stackId, z);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return bounds;
    }

    @Path("project/{project}/stack/{stack}/z/{z}/tileBounds")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<TileBounds> getTileBounds(@PathParam("owner") final String owner,
                                          @PathParam("project") final String project,
                                          @PathParam("stack") final String stack,
                                          @PathParam("z") final Double z) {

        LOG.info("getTileBounds: entry, owner={}, project={}, stack={}, z={}",
                 owner, project, stack, z);

        List<TileBounds> list = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            list = renderDao.getTileBounds(stackId, z);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return list;
    }

    @Path("project/{project}/stack/{stack}/z/{z}/resolvedTiles")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public ResolvedTileSpecCollection getResolvedTiles(@PathParam("owner") final String owner,
                                                       @PathParam("project") final String project,
                                                       @PathParam("stack") final String stack,
                                                       @PathParam("z") final Double z) {

        LOG.info("getResolvedTiles: entry, owner={}, project={}, stack={}, z={}",
                 owner, project, stack, z);

        ResolvedTileSpecCollection resolvedTiles = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            resolvedTiles = renderDao.getResolvedTiles(stackId, z);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return resolvedTiles;
    }

    @Path("project/{project}/stack/{stack}/resolvedTiles")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public ResolvedTileSpecCollection getResolvedTiles(@PathParam("owner") final String owner,
                                                       @PathParam("project") final String project,
                                                       @PathParam("stack") final String stack,
                                                       @QueryParam("minZ") final Double minZ,
                                                       @QueryParam("maxZ") final Double maxZ,
                                                       @QueryParam("groupId") final String groupId,
                                                       @QueryParam("minX") final Double minX,
                                                       @QueryParam("maxX") final Double maxX,
                                                       @QueryParam("minY") final Double minY,
                                                       @QueryParam("maxY") final Double maxY) {

        LOG.info("getResolvedTiles: entry, owner={}, project={}, stack={}, minZ={}, maxZ={}, groupId={}, minX={}, maxX={}, minY={}, maxY={}",
                 owner, project, stack, minZ, maxZ, groupId, minX, maxX, minY, maxY);

        ResolvedTileSpecCollection resolvedTiles = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            resolvedTiles = renderDao.getResolvedTiles(stackId, minZ, maxZ, groupId, minX, maxX, minY, maxY);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return resolvedTiles;
    }

    @Path("project/{project}/stack/{stack}/resolvedTiles")
    @PUT
    @Consumes(MediaType.APPLICATION_JSON)
    public Response saveResolvedTiles(@PathParam("owner") final String owner,
                                      @PathParam("project") final String project,
                                      @PathParam("stack") final String stack,
                                      @Context final UriInfo uriInfo,
                                      final ResolvedTileSpecCollection resolvedTiles) {
        return saveResolvedTilesForZ(owner, project, stack, null, uriInfo, resolvedTiles);
    }

    @Path("project/{project}/stack/{stack}/z/{z}/resolvedTiles")
    @PUT
    @Consumes(MediaType.APPLICATION_JSON)
    public Response saveResolvedTilesForZ(@PathParam("owner") final String owner,
                                          @PathParam("project") final String project,
                                          @PathParam("stack") final String stack,
                                          @PathParam("z") final Double z,
                                          @Context final UriInfo uriInfo,
                                          final ResolvedTileSpecCollection resolvedTiles) {

        LOG.info("saveResolvedTilesForZ: entry, owner={}, project={}, stack={}, z={}",
                 owner, project, stack, z);

        try {
            if (resolvedTiles == null) {
                throw new IllegalServiceArgumentException("no resolved tiles provided");
            }

            final StackId stackId = new StackId(owner, project, stack);
            final StackMetaData stackMetaData = getStackMetaData(stackId);

            if (! stackMetaData.isLoading()) {
                throw new IllegalStateException("Resolved tiles can only be saved to stacks in the " +
                                                LOADING + " state, but this stack's state is " +
                                                stackMetaData.getState() + ".");
            }

            resolvedTiles.validateCollection(z);

            renderDao.saveResolvedTiles(stackId, resolvedTiles);

        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        final Response.ResponseBuilder responseBuilder = Response.created(uriInfo.getRequestUri());

        LOG.info("saveResolvedTilesForZ: exit");

        return responseBuilder.build();
    }

    @Path("project/{project}/stack/{stack}/section/{sectionId}/z")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Double getZForSection(@PathParam("owner") final String owner,
                                 @PathParam("project") final String project,
                                 @PathParam("stack") final String stack,
                                 @PathParam("sectionId") final String sectionId,
                                 @Context final UriInfo uriInfo) {

        LOG.info("getZForSection: entry, owner={}, project={}, stack={}, sectionId={}",
                 owner, project, stack, sectionId);

        Double z = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            z = renderDao.getZForSection(stackId, sectionId);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return z;
    }

    @Path("project/{project}/stack/{stack}/section/{sectionId}/z")
    @PUT
    @Consumes(MediaType.APPLICATION_JSON)
    public Response updateZForSection(@PathParam("owner") final String owner,
                                      @PathParam("project") final String project,
                                      @PathParam("stack") final String stack,
                                      @PathParam("sectionId") final String sectionId,
                                      @Context final UriInfo uriInfo,
                                      final Double z) {
        LOG.info("updateZForSection: entry, owner={}, project={}, stack={}, sectionId={}, z={}",
                 owner, project, stack, sectionId, z);

        try {
            final StackId stackId = new StackId(owner, project, stack);
            final StackMetaData stackMetaData = getStackMetaData(stackId);

            if (! stackMetaData.isLoading()) {
                throw new IllegalStateException("Z values can only be updated for stacks in the " +
                                                LOADING + " state, but this stack's state is " +
                                                stackMetaData.getState() + ".");
            }

            renderDao.updateZForSection(stackId, sectionId, z);

        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        final Response.ResponseBuilder responseBuilder = Response.created(uriInfo.getRequestUri());

        LOG.info("updateZForSection: exit");

        return responseBuilder.build();
    }

    @Path("project/{project}/stack/{stack}/tile/{tileId}")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public TileSpec getTileSpec(@PathParam("owner") final String owner,
                                @PathParam("project") final String project,
                                @PathParam("stack") final String stack,
                                @PathParam("tileId") final String tileId) {

        LOG.info("getTileSpec: entry, owner={}, project={}, stack={}, tileId={}",
                 owner, project, stack, tileId);

        TileSpec tileSpec = null;
        try {
            tileSpec = getTileSpec(owner, project, stack, tileId, false);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        return tileSpec;
    }

    @Path("project/{project}/stack/{stack}/tile/{tileId}/render-parameters")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RenderParameters getRenderParameters(@PathParam("owner") final String owner,
                                                @PathParam("project") final String project,
                                                @PathParam("stack") final String stack,
                                                @PathParam("tileId") final String tileId,
                                                @QueryParam("scale") Double scale,
                                                @QueryParam("filter") final Boolean filter,
                                                @QueryParam("binaryMask") final Boolean binaryMask) {

        LOG.info("getRenderParameters: entry, owner={}, project={}, stack={}, tileId={}, scale={}, filter={}, binaryMask={}",
                 owner, project, stack, tileId, scale, filter, binaryMask);

        RenderParameters parameters = null;
        try {
            final TileSpec tileSpec = getTileSpec(owner, project, stack, tileId, true);
            tileSpec.flattenTransforms();
            if (scale == null) {
                scale = 1.0;
            }

            final StackId stackId = new StackId(owner, project, stack);
            final StackMetaData stackMetaData = getStackMetaData(stackId);

            final Integer stackLayoutWidth = stackMetaData.getLayoutWidth();
            final Integer stackLayoutHeight = stackMetaData.getLayoutHeight();

            final int margin = 0;
            final Double x = getLayoutMinValue(tileSpec.getMinX(), margin);
            final Double y = getLayoutMinValue(tileSpec.getMinY(), margin);
            final Integer width = getLayoutSizeValue(stackLayoutWidth, tileSpec.getWidth(), margin);
            final Integer height = getLayoutSizeValue(stackLayoutHeight, tileSpec.getHeight(), margin);

            parameters = new RenderParameters(null, x, y, width, height, scale);
            parameters.setDoFilter(filter);
            parameters.setBinaryMask(binaryMask);
            parameters.addTileSpec(tileSpec);
            parameters.setMipmapPathBuilder(stackMetaData.getCurrentMipmapPathBuilder());

        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        return parameters;
    }

    @Path("project/{project}/stack/{stack}/tile/{tileId}/source/scale/{scale}/render-parameters")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RenderParameters getTileSourceRenderParameters(@PathParam("owner") final String owner,
                                                          @PathParam("project") final String project,
                                                          @PathParam("stack") final String stack,
                                                          @PathParam("tileId") final String tileId,
                                                          @PathParam("scale") final Double scale,
                                                          @QueryParam("filter") final Boolean filter) {

        LOG.info("getTileSourceRenderParameters: entry, owner={}, project={}, stack={}, tileId={}, scale={}, filter={}",
                 owner, project, stack, tileId, scale, filter);

        return getTileRenderParameters(owner, project, stack, tileId, scale, filter, true);
    }

    @Path("project/{project}/stack/{stack}/tile/{tileId}/mask/scale/{scale}/render-parameters")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RenderParameters getTileMaskRenderParameters(@PathParam("owner") final String owner,
                                                        @PathParam("project") final String project,
                                                        @PathParam("stack") final String stack,
                                                        @PathParam("tileId") final String tileId,
                                                        @PathParam("scale") final Double scale,
                                                        @QueryParam("filter") final Boolean filter) {

        LOG.info("getTileMaskRenderParameters: entry, owner={}, project={}, stack={}, tileId={}, scale={}, filter={}",
                 owner, project, stack, tileId, scale, filter);

        return getTileRenderParameters(owner, project, stack, tileId, scale, filter, false);
    }

    @Path("project/{project}/stack/{stack}/transform/{transformId}")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public TransformSpec getTransformSpec(@PathParam("owner") final String owner,
                                          @PathParam("project") final String project,
                                          @PathParam("stack") final String stack,
                                          @PathParam("transformId") final String transformId) {

        LOG.info("getTransformSpec: entry, owner={}, project={}, stack={}, transformId={}",
                 owner, project, stack, transformId);

        TransformSpec transformSpec = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            transformSpec = renderDao.getTransformSpec(stackId, transformId);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        return transformSpec;
    }

    /**
     * @return number of tiles within the specified bounding box.
     */
    @Path("project/{project}/stack/{stack}/z/{z}/box/{x},{y},{width},{height}/tile-count")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Long getTileCount(@PathParam("owner") final String owner,
                             @PathParam("project") final String project,
                             @PathParam("stack") final String stack,
                             @PathParam("x") final Double x,
                             @PathParam("y") final Double y,
                             @PathParam("z") final Double z,
                             @PathParam("width") final Integer width,
                             @PathParam("height") final Integer height) {

        LOG.info("getTileCount: entry, owner={}, project={}, stack={}, x={}, y={}, z={}, width={}, height={}",
                 owner, project, stack, x, y, z, width, height);

        long count = 0;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            count = renderDao.getTileCount(stackId, x, y, z, width, height);
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return count;
    }

    /**
     * @return list of tile specs for specified layer with flattened (and therefore resolved)
     *         transform specs suitable for external use.
     */
    @Path("project/{project}/stack/{stack}/z/{z}/tile-specs")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public List<TileSpec> getTileSpecsForZ(@PathParam("owner") final String owner,
                                           @PathParam("project") final String project,
                                           @PathParam("stack") final String stack,
                                           @PathParam("z") final Double z) {

        LOG.info("getTileSpecsForZ: entry, owner={}, project={}, stack={}, z={}",
                 owner, project, stack, z);

        List<TileSpec> tileSpecList = null;
        try {
            final RenderParameters parameters = getRenderParametersForZ(owner, project, stack, z, 1.0, false);
            tileSpecList = parameters.getTileSpecs();
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        return tileSpecList;
    }

    /**
     * @return render parameters for specified layer with flattened (and therefore resolved)
     *         transform specs suitable for external use.
     */
    @Path("project/{project}/stack/{stack}/z/{z}/render-parameters")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RenderParameters getRenderParametersForZ(@PathParam("owner") final String owner,
                                                    @PathParam("project") final String project,
                                                    @PathParam("stack") final String stack,
                                                    @PathParam("z") final Double z,
                                                    @QueryParam("scale") final Double scale,
                                                    @QueryParam("filter") final Boolean filter) {

        LOG.info("getRenderParametersForZ: entry, owner={}, project={}, stack={}, z={}, scale={}",
                 owner, project, stack, z, scale);

        RenderParameters parameters = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            parameters = renderDao.getParameters(stackId, z, scale);
            parameters.setDoFilter(filter);

            final StackMetaData stackMetaData = getStackMetaData(stackId);
            final MipmapPathBuilder mipmapPathBuilder = stackMetaData.getCurrentMipmapPathBuilder();
            if (mipmapPathBuilder != null) {
                parameters.setMipmapPathBuilder(mipmapPathBuilder);
            }
            parameters.flattenTransforms();
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }

        return parameters;
    }

    /**
     * @return render parameters for specified bounding box with flattened (and therefore resolved)
     *         transform specs suitable for external use.
     */
    @Path("project/{project}/stack/{stack}/z/{z}/box/{x},{y},{width},{height},{scale}/render-parameters")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RenderParameters getExternalRenderParameters(@PathParam("owner") final String owner,
                                                        @PathParam("project") final String project,
                                                        @PathParam("stack") final String stack,
                                                        @PathParam("x") final Double x,
                                                        @PathParam("y") final Double y,
                                                        @PathParam("z") final Double z,
                                                        @PathParam("width") final Integer width,
                                                        @PathParam("height") final Integer height,
                                                        @PathParam("scale") final Double scale) {

        return getExternalRenderParameters(owner, project, stack, null, x, y, z, width, height, scale);
    }

    /**
     * @return render parameters for specified bounding box with flattened (and therefore resolved)
     *         transform specs suitable for external use.
     */
    @Path("project/{project}/stack/{stack}/group/{groupId}/z/{z}/box/{x},{y},{width},{height},{scale}/render-parameters")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RenderParameters getExternalRenderParameters(@PathParam("owner") final String owner,
                                                        @PathParam("project") final String project,
                                                        @PathParam("stack") final String stack,
                                                        @PathParam("groupId") final String groupId,
                                                        @PathParam("x") final Double x,
                                                        @PathParam("y") final Double y,
                                                        @PathParam("z") final Double z,
                                                        @PathParam("width") final Integer width,
                                                        @PathParam("height") final Integer height,
                                                        @PathParam("scale") final Double scale) {

        LOG.info("getExternalRenderParameters: entry, owner={}, project={}, stack={}, groupId={}, x={}, y={}, z={}, width={}, height={}, scale={}",
                 owner, project, stack, groupId, x, y, z, width, height, scale);

        RenderParameters parameters = null;
        try {
            final StackId stackId = new StackId(owner, project, stack);
            parameters = getInternalRenderParameters(stackId, groupId, x, y, z, width, height, scale);
            parameters.flattenTransforms();
        } catch (final Throwable t) {
            RenderServiceUtil.throwServiceException(t);
        }
        return parameters;
    }

    /**
     * @return render parameters for specified bounding box with in-memory resolved
     *         transform specs suitable for internal use.
     */
    public RenderParameters getInternalRenderParameters(final StackId stackId,
                                                        final String groupId,
                                                        final Double x,
                                                        final Double y,
                                                        final Double z,
                                                        final Integer width,
                                                        final Integer height,
                                                        final Double scale) {


        final RenderParameters parameters = renderDao.getParameters(stackId, groupId, x, y, z, width, height, scale);
        final StackMetaData stackMetaData = getStackMetaData(stackId);
        final MipmapPathBuilder mipmapPathBuilder = stackMetaData.getCurrentMipmapPathBuilder();
        if (mipmapPathBuilder != null) {
            parameters.setMipmapPathBuilder(mipmapPathBuilder);
        }
        return parameters;
    }

    private TileSpec getTileSpec(final String owner,
                                 final String project,
                                 final String stack,
                                 final String tileId,
                                 final boolean resolveTransformReferences) {
        final StackId stackId = new StackId(owner, project, stack);
        return renderDao.getTileSpec(stackId, tileId, resolveTransformReferences);
    }

    private RenderParameters getTileRenderParameters(final String owner,
                                                     final String project,
                                                     final String stack,
                                                     final String tileId,
                                                     final Double scale,
                                                     final Boolean filter,
                                                     final boolean isSource) {

        // we only need to fetch the tile spec since no transforms are needed
        final TileSpec tileSpec = getTileSpec(owner, project, stack, tileId);

        final RenderParameters tileRenderParameters =
                new RenderParameters(null, 0, 0, tileSpec.getWidth(), tileSpec.getHeight(), scale);

        final Map.Entry<Integer, ImageAndMask> firstEntry = tileSpec.getFirstMipmapEntry();
        final ImageAndMask imageAndMask = firstEntry.getValue();
        final TileSpec simpleTileSpec = new TileSpec();
        if (isSource) {
            final ImageAndMask imageWithoutMask = new ImageAndMask(imageAndMask.getImageUrl(), null);
            simpleTileSpec.putMipmap(firstEntry.getKey(), imageWithoutMask);
        } else {
            final ImageAndMask maskAsImage = new ImageAndMask(imageAndMask.getMaskUrl(), null);
            simpleTileSpec.putMipmap(firstEntry.getKey(), maskAsImage);
        }
        tileRenderParameters.addTileSpec(simpleTileSpec);
        tileRenderParameters.setDoFilter(filter);

        return tileRenderParameters;
    }

    private Double getLayoutMinValue(final Double minValue,
                                     final int margin) {
        Double layoutValue = null;
        if (minValue != null) {
            layoutValue = minValue - margin;
        }
        return layoutValue;
    }

    private Integer getLayoutSizeValue(final Integer stackValue,
                                       final Integer tileValue,
                                       final int margin) {
        Integer layoutValue = null;

        if (stackValue != null) {
            layoutValue = stackValue + margin;
        } else if ((tileValue != null) && (tileValue >= 0)) {
            layoutValue = tileValue + margin;
        }

        if ((layoutValue != null) && ((layoutValue % 2) != 0)) {
            layoutValue = layoutValue + 1;
        }

        return layoutValue;
    }

    private StackMetaData getStackMetaData(final StackId stackId) {
        final StackMetaData stackMetaData = renderDao.getStackMetaData(stackId);
        if (stackMetaData == null) {
            throw StackMetaDataService.getStackNotFoundException(stackId.getOwner(),
                                                                 stackId.getProject(),
                                                                 stackId.getStack());
        }
        return stackMetaData;
    }

    private static final Logger LOG = LoggerFactory.getLogger(RenderDataService.class);
}
