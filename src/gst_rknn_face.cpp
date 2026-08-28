#include "face_tracker.h"
#include "gst_rknn_face_meta.h"

#include <gst/allocators/gstdmabuf.h>
#include <gst/video/gstvideofilter.h>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>

#include <cerrno>
#include <cstring>
#include <exception>

#ifndef PACKAGE
#define PACKAGE "rknn-face-points"
#endif

extern "C" {

static gboolean rknn_face_meta_init(GstMeta* meta, gpointer, GstBuffer*) {
  auto* face = reinterpret_cast<GstRknnFaceMeta*>(meta);
  face->found = FALSE;
  face->landmark_count = 0;
  face->score = 0;
  face->tongue_out = 0;
  face->npu_ms = 0;
  std::memset(face->bbox, 0, sizeof(face->bbox));
  std::memset(face->head_pose_deg, 0, sizeof(face->head_pose_deg));
  std::memset(face->landmarks, 0, sizeof(face->landmarks));
  std::memset(face->blendshapes, 0, sizeof(face->blendshapes));
  return TRUE;
}

static gboolean rknn_face_meta_transform(GstBuffer* dest, GstMeta* meta, GstBuffer*, GQuark, gpointer) {
  auto* src = reinterpret_cast<GstRknnFaceMeta*>(meta);
  auto* out = gst_buffer_add_rknn_face_meta(dest);
  out->found = src->found;
  out->landmark_count = src->landmark_count;
  out->score = src->score;
  out->tongue_out = src->tongue_out;
  out->npu_ms = src->npu_ms;
  std::memcpy(out->bbox, src->bbox, sizeof(src->bbox));
  std::memcpy(out->head_pose_deg, src->head_pose_deg, sizeof(src->head_pose_deg));
  std::memcpy(out->landmarks, src->landmarks, sizeof(src->landmarks));
  std::memcpy(out->blendshapes, src->blendshapes, sizeof(src->blendshapes));
  return TRUE;
}

GType gst_rknn_face_meta_api_get_type(void) {
  static gsize type = 0;
  static const gchar* tags[] = {GST_META_TAG_VIDEO_STR, nullptr};
  if (g_once_init_enter(&type)) {
    GType value = gst_meta_api_type_register("GstRknnFaceMetaAPI", tags);
    g_once_init_leave(&type, value);
  }
  return static_cast<GType>(type);
}

const GstMetaInfo* gst_rknn_face_meta_get_info(void) {
  static gsize info = 0;
  if (g_once_init_enter(&info)) {
    const GstMetaInfo* value = gst_meta_register(
        GST_RKNN_FACE_META_API_TYPE, "GstRknnFaceMeta", sizeof(GstRknnFaceMeta),
        rknn_face_meta_init, nullptr, rknn_face_meta_transform);
    g_once_init_leave(&info, reinterpret_cast<gsize>(value));
  }
  return reinterpret_cast<const GstMetaInfo*>(info);
}

GstRknnFaceMeta* gst_buffer_add_rknn_face_meta(GstBuffer* buffer) {
  return reinterpret_cast<GstRknnFaceMeta*>(gst_buffer_add_meta(buffer, GST_RKNN_FACE_META_INFO, nullptr));
}

}  // extern "C"

typedef struct _GstRknnFaceMesh {
  GstVideoFilter parent;
  gchar* model_dir;
  gboolean draw;
  guint inference_interval;
  guint64 frame_number;
  gboolean last_found;
  rknn_face::Tracker* tracker;
  rknn_face::Result* last_result;
} GstRknnFaceMesh;

typedef struct _GstRknnFaceMeshClass {
  GstVideoFilterClass parent_class;
} GstRknnFaceMeshClass;

#define GST_TYPE_RKNN_FACE_MESH (gst_rknn_face_mesh_get_type())
#define GST_RKNN_FACE_MESH(obj) (reinterpret_cast<GstRknnFaceMesh*>(obj))

G_DEFINE_TYPE(GstRknnFaceMesh, gst_rknn_face_mesh, GST_TYPE_VIDEO_FILTER)

enum {
  PROP_0,
  PROP_MODEL_DIR,
  PROP_DRAW,
  PROP_INFERENCE_INTERVAL,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=(string)NV12"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=(string)NV12"));

static gboolean sync_dmabufs(GstBuffer* buffer, guint64 flags, gchar** error) {
  guint count = gst_buffer_n_memory(buffer);
  if (!count) {
    *error = g_strdup("buffer has no memory");
    return FALSE;
  }
  for (guint i = 0; i < count; ++i) {
    GstMemory* memory = gst_buffer_peek_memory(buffer, i);
    if (!gst_is_dmabuf_memory(memory)) {
      *error = g_strdup_printf("memory %u is not DMABUF", i);
      return FALSE;
    }
    dma_buf_sync sync{flags};
    int fd = gst_dmabuf_memory_get_fd(memory);
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) < 0) {
      *error = g_strdup_printf("DMABUF fd %d sync failed: %s", fd, g_strerror(errno));
      return FALSE;
    }
  }
  return TRUE;
}

static void gst_rknn_face_mesh_set_property(GObject* object, guint id,
                                            const GValue* value, GParamSpec* spec) {
  auto* self = GST_RKNN_FACE_MESH(object);
  switch (id) {
    case PROP_MODEL_DIR:
      g_free(self->model_dir);
      self->model_dir = g_value_dup_string(value);
      break;
    case PROP_DRAW:
      self->draw = g_value_get_boolean(value);
      break;
    case PROP_INFERENCE_INTERVAL:
      self->inference_interval = g_value_get_uint(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}

static void gst_rknn_face_mesh_get_property(GObject* object, guint id,
                                            GValue* value, GParamSpec* spec) {
  auto* self = GST_RKNN_FACE_MESH(object);
  switch (id) {
    case PROP_MODEL_DIR:
      g_value_set_string(value, self->model_dir);
      break;
    case PROP_DRAW:
      g_value_set_boolean(value, self->draw);
      break;
    case PROP_INFERENCE_INTERVAL:
      g_value_set_uint(value, self->inference_interval);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}

static gboolean gst_rknn_face_mesh_start(GstBaseTransform* transform) {
  auto* self = GST_RKNN_FACE_MESH(transform);
  try {
    self->tracker = new rknn_face::Tracker(self->model_dir);
    self->last_result = new rknn_face::Result;
    self->frame_number = 0;
    return TRUE;
  } catch (const std::exception& e) {
    delete self->tracker;
    self->tracker = nullptr;
    GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ, ("failed to load RKNN models"), ("%s", e.what()));
    return FALSE;
  }
}

static gboolean gst_rknn_face_mesh_stop(GstBaseTransform* transform) {
  auto* self = GST_RKNN_FACE_MESH(transform);
  delete self->tracker;
  delete self->last_result;
  self->tracker = nullptr;
  self->last_result = nullptr;
  return TRUE;
}

static GstFlowReturn gst_rknn_face_mesh_transform_frame_ip(GstVideoFilter* filter,
                                                           GstVideoFrame* frame) {
  auto* self = GST_RKNN_FACE_MESH(filter);
  int width = GST_VIDEO_FRAME_WIDTH(frame);
  int height = GST_VIDEO_FRAME_HEIGHT(frame);
  rknn_face::Nv12Image image{
      width, height,
      GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0), GST_VIDEO_FRAME_PLANE_STRIDE(frame, 1),
      static_cast<unsigned char*>(GST_VIDEO_FRAME_PLANE_DATA(frame, 0)),
      static_cast<unsigned char*>(GST_VIDEO_FRAME_PLANE_DATA(frame, 1))};
  if (GST_VIDEO_FRAME_N_PLANES(frame) != 2 || image.y_stride < width || image.uv_stride < width) {
    GST_ELEMENT_ERROR(self, STREAM, FORMAT, ("invalid NV12 frame"),
                      ("planes=%u strides=%d,%d width=%d",
                       GST_VIDEO_FRAME_N_PLANES(frame), image.y_stride, image.uv_stride, width));
    return GST_FLOW_ERROR;
  }

  gchar* sync_error = nullptr;
  if (!sync_dmabufs(frame->buffer, DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW, &sync_error)) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("DMABUF zero-copy input required"), ("%s", sync_error));
    g_free(sync_error);
    return GST_FLOW_ERROR;
  }

  gboolean inferred = self->frame_number++ % self->inference_interval == 0;
  try {
    if (inferred) self->last_found = self->tracker->process(image, *self->last_result);
    if (self->last_found && self->draw) rknn_face::draw(image, *self->last_result);
  } catch (const std::exception& e) {
    sync_dmabufs(frame->buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW, &sync_error);
    g_free(sync_error);
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("RKNN inference failed"), ("%s", e.what()));
    return GST_FLOW_ERROR;
  }

  if (!sync_dmabufs(frame->buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW, &sync_error)) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("DMABUF zero-copy output sync failed"), ("%s", sync_error));
    g_free(sync_error);
    return GST_FLOW_ERROR;
  }

  auto* meta = gst_buffer_add_rknn_face_meta(frame->buffer);
  const auto& result = *self->last_result;
  meta->found = self->last_found;
  meta->score = result.score;
  meta->npu_ms = inferred ? result.detector_ms + result.landmark_ms + result.blendshape_ms : 0;
  if (self->last_found) {
    meta->landmark_count = static_cast<guint>(result.landmarks.size());
    meta->bbox[0] = result.bbox_x;
    meta->bbox[1] = result.bbox_y;
    meta->bbox[2] = result.bbox_w;
    meta->bbox[3] = result.bbox_h;
    meta->tongue_out = result.tongue_out;
    std::memcpy(meta->head_pose_deg, result.head_pose_deg.data(), sizeof(meta->head_pose_deg));
    std::memcpy(meta->blendshapes, result.blendshapes.data(), sizeof(meta->blendshapes));
    for (size_t i = 0; i < result.landmarks.size(); ++i) {
      meta->landmarks[i][0] = result.landmarks[i].x;
      meta->landmarks[i][1] = result.landmarks[i].y;
      meta->landmarks[i][2] = result.landmarks[i].z;
    }
  }
  return GST_FLOW_OK;
}

static void gst_rknn_face_mesh_finalize(GObject* object) {
  auto* self = GST_RKNN_FACE_MESH(object);
  delete self->tracker;
  delete self->last_result;
  g_free(self->model_dir);
  G_OBJECT_CLASS(gst_rknn_face_mesh_parent_class)->finalize(object);
}

static void gst_rknn_face_mesh_class_init(GstRknnFaceMeshClass* klass) {
  auto* object_class = G_OBJECT_CLASS(klass);
  auto* element_class = GST_ELEMENT_CLASS(klass);
  auto* transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  auto* video_class = GST_VIDEO_FILTER_CLASS(klass);
  object_class->set_property = gst_rknn_face_mesh_set_property;
  object_class->get_property = gst_rknn_face_mesh_get_property;
  object_class->finalize = gst_rknn_face_mesh_finalize;
  g_object_class_install_property(object_class, PROP_MODEL_DIR,
      g_param_spec_string("model-dir", "Model directory", "Directory containing the three RKNN models",
                          "/app/models", static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(object_class, PROP_DRAW,
      g_param_spec_boolean("draw", "Draw landmarks", "Draw the face box and 478 landmarks on output frames",
                           TRUE, static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(object_class, PROP_INFERENCE_INTERVAL,
      g_param_spec_uint("inference-interval", "Inference interval",
                        "Run RKNN every N frames and reuse the last result between runs",
                        1, 60, 2, static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  gst_element_class_set_static_metadata(element_class, "RKNN face mesh", "Filter/Video",
      "RK3588 face detection, 478 landmarks, head pose and blendshapes", "CmST0us");
  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);
  transform_class->start = gst_rknn_face_mesh_start;
  transform_class->stop = gst_rknn_face_mesh_stop;
  video_class->transform_frame_ip = gst_rknn_face_mesh_transform_frame_ip;
}

static void gst_rknn_face_mesh_init(GstRknnFaceMesh* self) {
  self->model_dir = g_strdup("/app/models");
  self->draw = TRUE;
  // ponytail: 30 Hz inference keeps 60 fps media real-time; set interval=1 when every frame must be fresh.
  self->inference_interval = 2;
  self->frame_number = 0;
  self->last_found = FALSE;
  self->tracker = nullptr;
  self->last_result = nullptr;
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "rknnfacemesh", GST_RANK_NONE, GST_TYPE_RKNN_FACE_MESH);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, rknnface,
                  "RKNN face mesh filter", plugin_init, "1.0.0", "LGPL",
                  "rknn-face-points", "https://github.com/CmST0us/rknn-face-points")
