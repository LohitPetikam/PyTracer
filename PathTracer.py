import core_utils as ut
import numpy as np
from glumpy import app, gloo, gl, data, __version__
import imageio # to read exr files

class PathTracerProgram(object):
	def __init__(self, resolution, env_image):
		super(PathTracerProgram, self).__init__()
		self.resolution = resolution;
		self.env_image = env_image;
		self.init()

	def init(self):
		vertex = """
			attribute vec2 position;
			void main() { gl_Position = vec4(position, 0.0, 1.0); }
		"""
		fragment =  ut.read_file('path_tracer.frag')
		self.quad = gloo.Program(vertex, fragment, count=4)
		self.quad['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
		self.quad['u_Resolution'] = self.resolution

		rgb = self.env_image
		print(np.min(rgb), np.max(rgb))
		self.quad['u_EnvTexture'] = np.asarray(np.flip(np.flip(rgb, axis=0), axis=1)).copy().view(gloo.TextureFloat2D)
		self.quad['u_EnvTexture'].interpolation = (gl.GL_LINEAR, gl.GL_LINEAR)

		self.num_samples = 0
		self.quad['u_NumSamples'] = self.num_samples

		self.quad['u_Rand'] = np.random.uniform(0, 1, 4)

		n = np.dstack((
			np.float32(np.random.uniform(0, 1, self.resolution[::-1])),
			np.float32(np.random.uniform(0, 1, self.resolution[::-1])),
			np.float32(np.random.uniform(0, 1, self.resolution[::-1]))
			))
		self.noise_texture = n.view(gloo.TextureFloat2D)
		self.noise_texture.interpolation = gl.GL_LINEAR
		self.noise_texture.wrapping = gl.GL_REPEAT
		self.noise_texture.gpu_format = gl.GL_RGB32F
		self.quad['u_Noise'] = self.noise_texture

		self.render_texture = np.zeros(self.resolution[::-1],np.float32).view(gloo.TextureFloat2D)
		self.render_texture.interpolation = gl.GL_NEAREST
		self.render_texture.wrapping = gl.GL_REPEAT
		self.render_texture.gpu_format = gl.GL_RGB32F
		self.render_fbo = gloo.FrameBuffer(color=self.render_texture)

		self.quad['u_Texture'] = self.render_texture

	def UpdateCamera(self, pitch, yaw, zoom, ez):
		self.quad['u_Pitch'] = pitch
		self.quad['u_Yaw'] = yaw
		self.quad['u_Zoom'] = zoom
		self.quad['u_Ez'] = ez
		self.num_samples = 0
		self.quad['u_NumSamples'] = self.num_samples

	def RenderFBO(self):
		self.render_fbo.activate()
		gl.glViewport(0, 0, self.resolution[0], self.resolution[1])
		self.quad.draw(gl.GL_TRIANGLE_STRIP)
		self.render_fbo.deactivate()
		self.num_samples += 1
		self.quad['u_NumSamples'] = self.num_samples
		self.quad['u_Rand'] = np.random.uniform(0, 1, 4)

	def ReadPixels(self, x, y, w, h):
		self.render_fbo.activate()
		img = gl.glReadPixels(x, y, w, h, gl.GL_RGB, gl.GL_FLOAT)
		self.render_fbo.deactivate()
		return img

	def GetTexture(self):
		return self.render_texture

	def GetNumSamples(self):
		return self.num_samples

class OutputProgram(object):
	def __init__(self):
		super(OutputProgram, self).__init__()
		self.input_texture = np.zeros((5,5),np.float32).view(gloo.TextureFloat2D);
		self.init()

	def init(self):
		
		vertex = """
			attribute vec2 position;
			varying vec2 vUV;
			void main() { vUV=0.5+0.5*position; gl_Position = vec4(position, 0.0, 1.0); }
		"""
		fragment =  """
		uniform sampler2D u_Texture;
		uniform float u_Gain = 1.0;
		varying vec2 vUV;
		vec3 ACESFilm(vec3 x) {
			float a = 2.51;
			float b = 0.03;
			float c = 2.43;
			float d = 0.59;
			float e = 0.14;
			return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0, 1);
		}
		void main() {
				vec2 uv = vUV;
				vec3 out_col = texture2D(u_Texture, uv).xyz;
				gl_FragColor = vec4(ACESFilm(u_Gain*out_col), 1.0);
		}
		"""
		self.quad = gloo.Program(vertex, fragment, count=4)
		self.quad['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
		self.quad['u_Texture'] = self.input_texture

	def Render(self):
		self.quad.draw(gl.GL_TRIANGLE_STRIP)

	def SetTexture(self, texture):
		self.input_texture = texture
		self.quad['u_Texture'] = self.input_texture

	def SetGain(self, g):
		self.quad['u_Gain'] = g

class Viewer(object):
	def __init__(self):
		super(Viewer, self).__init__()
		self.init()

	def init_path_tracer(self):
		self.render_program = PathTracerProgram(self.window.get_size(), self.env_image)
		self.render_program.UpdateCamera(self.cam_pitch, self.cam_yaw, self.cam_zoom, self.cam_ez)
		self.output_program.SetTexture(self.render_program.GetTexture())

	def init(self):

		self.console = app.Console(rows=32,cols=80, scale=2, color=(1,1,1,1))
		self.window = app.Window(640,480,color=(1,0,1,1))

		self.show_console = True
		
		self.gain = 1

		self.cam_pitch = -1
		self.cam_yaw = 0
		self.cam_zoom = 3
		self.cam_ez = 1

		self.mouse_colour = 0

		self.env_image = imageio.imread(ut.full_path('../TestRMs/00391_OpenfootageNET_fieldanif_low.exr'), 'EXR-FI')

		@self.window.event
		def on_init():
			self.window.activate()

			self.output_program = OutputProgram()
			self.output_program.SetGain(self.gain)

			self.init_path_tracer()

			self.window.attach(self.console)

		@self.window.event
		def on_resize(width,height):
			self.window.activate()
			self.init_path_tracer()

		@self.window.event
		def on_mouse_drag(x, y, dx, dy, button):
			self.window.activate()
			# print('Mouse drag (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f, button=%d)' % (x,y,dx,dy,button))
			s = 0.01
			if button == 2: # LMB
				self.cam_pitch = np.max([ -np.pi/2, np.min([np.pi/2, self.cam_pitch - s*dy]) ])
				self.cam_yaw = self.cam_yaw - s*dx
				self.render_program.UpdateCamera(self.cam_pitch, self.cam_yaw, self.cam_zoom, self.cam_ez)
			if button == 8: # RMB
				self.gain *= np.arctan(0.2*dx)/np.pi+1
				self.output_program.SetGain(self.gain)

		@self.window.event
		def on_mouse_scroll(x, y, dx, dy):
			self.window.activate()
			# print('Mouse scroll (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy))
			self.cam_zoom *= np.arctan(-0.2*dy)/np.pi+1
			self.render_program.UpdateCamera(self.cam_pitch, self.cam_yaw, self.cam_zoom, self.cam_ez)

		@self.window.event
		def on_mouse_press(x, y, button):
			self.window.activate()
			# print('Mouse button pressed (x=%.1f, y=%.1f, button=%d)' % (x,y,button))
			if button == 8: # RMB
				pass
			elif button == 4: # MMB
				pass

		@self.window.event
		def on_mouse_motion(x, y, dx, dy):
			# print('Mouse motion (x=%.1f, y=%.1f, dx=%.1f, dy=%.1f)' % (x,y,dx,dy))
			self.mouse_colour = self.render_program.ReadPixels(x, self.window.height-y, 1, 1)

		@self.window.timer(1/30.0)
		def timer(dt):
			self.console.clear()
			self.console.write(" ")
			self.console.write(" FPS: %.2f (%.2f ms)" % (self.window.fps, 1000.0/(1.0e-6+self.window.fps)))
			self.console.write(" Samples: %d" % (self.render_program.GetNumSamples()))
			self.console.write(" Gain: %.2f" % (self.gain))
			self.console.write(" Mouse colour: {}".format(self.mouse_colour))

		@self.window.event
		def on_draw(dt):
			self.window.activate()

			self.render_program.RenderFBO()

			gl.glViewport(0, 0, self.window.width, self.window.height)
			self.output_program.Render()

			if self.show_console:
				self.console.draw()

		@self.window.event
		def on_key_press(symbol, modifiers):
			# print('Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers))
			ctrl_down = (modifiers==2)
			shift_down = (modifiers==1)

			if symbol == 65289: # tab
				pass
			elif ctrl_down and symbol==83: # ctrl+s
				pass
			elif ctrl_down and symbol==82: # ctrl+r
				self.init_path_tracer()
				print("shader reloaded")
			elif symbol==96: # tilde
				self.show_console = not self.show_console

		@self.window.event
		def on_close():
			pass

def main():
	v = Viewer()
	app.run(framerate=0)

if __name__ == '__main__':
	main()
